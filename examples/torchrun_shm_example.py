"""
使用 torchrun 启动时的共享内存适配方案示例

启动方式:
    torchrun --nproc_per_node=2 --master_port=29500 examples/torchrun_shm_example.py

主要变化:
1. 从环境变量获取 rank/world_size
2. 使用标准的 dist.init_process_group()（不手动传参数）
3. Event 同步改用 torch.multiprocessing 或 dist 原语
"""

import os
import torch
import pickle
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

# 方案 1: 使用 torch.multiprocessing.Event（推荐）
# 与原生 multiprocessing.Event 兼容，适合跨进程场景
import torch.multiprocessing as mp


class ModelRunnerWithTorchrun:
    """适配 torchrun 的 ModelRunner 示例"""
    
    def __init__(self, config, shm_name: str = "diffulex_shm", shm_size: int = 2**25):
        # 方案 A: 从环境变量获取分布式信息（torchrun 自动设置）
        if dist.is_initialized():
            # 如果已经初始化，直接使用
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # 从环境变量读取（torchrun 会设置这些）
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            # 初始化分布式环境
            dist.init_process_group(backend="nccl")
        
        self.config = config
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.shm: Optional[SharedMemory] = None
        
        # 设置设备（torchrun 设置 LOCAL_RANK）
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        torch.cuda.set_device(local_rank)
        
        # 初始化共享内存（逻辑与原来相同）
        if self.world_size > 1:
            self._init_shared_memory()
    
    def _init_shared_memory(self):
        """初始化共享内存 - 与原来逻辑相同"""
        if self.rank == 0:
            # rank 0 创建共享内存
            try:
                # 清理可能残留的共享内存
                old_shm = SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            
            self.shm = SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            # 等待其他进程准备好
            dist.barrier()
        else:
            # 其他进程等待 rank 0 创建完成
            dist.barrier()
            self.shm = SharedMemory(name=self.shm_name)
            # 启动工作循环
            self.loop()
    
    def write_shm(self, method_name: str, *args):
        """写入共享内存（rank 0 使用）"""
        assert self.world_size > 1 and self.rank == 0
        assert self.shm is not None
        
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        if n + 4 > self.shm_size:
            raise ValueError(
                f"Serialized data size ({n} bytes) exceeds shared memory buffer size "
                f"({self.shm_size} bytes)."
            )
        
        # 写入长度和数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        
        # 方案 1A: 使用 torch.distributed.broadcast 通知其他进程
        # 发送一个简单的通知标识
        notification = torch.tensor([1], dtype=torch.int32, device=f"cuda:{self.rank}")
        dist.broadcast(notification, src=0)
    
    def read_shm(self):
        """从共享内存读取（rank > 0 使用）"""
        assert self.world_size > 1 and self.rank > 0
        assert self.shm is not None
        
        # 方案 1A: 等待 rank 0 的广播通知
        notification = torch.tensor([0], dtype=torch.int32, device=f"cuda:{self.rank}")
        dist.broadcast(notification, src=0)
        
        # 读取数据
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        return method_name, args
    
    def loop(self):
        """工作进程循环"""
        while True:
            method_name, args = self.read_shm()
            method = getattr(self, method_name, None)
            if method:
                method(*args)
            if method_name == "exit":
                break
    
    def call(self, method_name: str, *args):
        """调用方法"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        
        # 所有进程都执行方法
        method = getattr(self, method_name, None)
        return method(*args) if method else None
    
    def exit(self):
        """清理资源"""
        if self.world_size > 1 and self.shm:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        
        if dist.is_initialized():
            dist.destroy_process_group()


# ============================================================================
# 方案 2: 使用 torch.multiprocessing.Event（如果需要在父进程中创建 Event）
# ============================================================================

class ModelRunnerWithTorchrunEvents:
    """使用 torch.multiprocessing.Event 的版本"""
    
    def __init__(self, config, shm_name: str = "diffulex_shm", 
                 shm_size: int = 2**25, events: Optional[list] = None):
        # 确保使用 torch.multiprocessing（spawn 模式）
        mp.set_start_method("spawn", force=True)
        
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            dist.init_process_group(backend="nccl")
        
        self.config = config
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.shm: Optional[SharedMemory] = None
        
        # Event 列表：每个 worker 进程对应一个 event
        # 注意：在 torchrun 场景下，这些 events 需要特殊处理
        # 因为 torchrun 创建的进程之间无法直接传递 Event 对象
        # 所以这里仍然使用 dist.broadcast 作为替代
        self.events = events
        
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        torch.cuda.set_device(local_rank)
        
        if self.world_size > 1:
            self._init_shared_memory()
    
    def _init_shared_memory(self):
        """初始化共享内存"""
        if self.rank == 0:
            try:
                old_shm = SharedMemory(name=self.shm_name)
                old_shm.close()
                old_shm.unlink()
            except FileNotFoundError:
                pass
            self.shm = SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
            dist.barrier()
        else:
            dist.barrier()
            self.shm = SharedMemory(name=self.shm_name)
            self.loop()
    
    def write_shm(self, method_name: str, *args):
        """写入共享内存"""
        assert self.world_size > 1 and self.rank == 0
        
        data = pickle.dumps([method_name, *args])
        n = len(data)
        
        if n + 4 > self.shm_size:
            raise ValueError(f"Data size exceeds shared memory buffer.")
        
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        
        # 使用 dist.broadcast 通知所有进程
        notification = torch.tensor([1], dtype=torch.int32, device=f"cuda:{self.rank}")
        dist.broadcast(notification, src=0)
    
    def read_shm(self):
        """读取共享内存"""
        assert self.world_size > 1 and self.rank > 0
        
        # 等待 rank 0 的广播
        notification = torch.tensor([0], dtype=torch.int32, device=f"cuda:{self.rank}")
        dist.broadcast(notification, src=0)
        
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        return method_name, args
    
    def loop(self):
        """工作进程循环"""
        while True:
            method_name, args = self.read_shm()
            method = getattr(self, method_name, None)
            if method:
                method(*args)
            if method_name == "exit":
                break


# ============================================================================
# 方案 3: 完全使用 torch.distributed 通信（不依赖共享内存）
# ============================================================================

class ModelRunnerWithDistComm:
    """完全不使用共享内存，改用 dist.broadcast_object_list"""
    
    def __init__(self, config):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            dist.init_process_group(backend="nccl")
        
        self.config = config
        local_rank = int(os.environ.get("LOCAL_RANK", str(self.rank)))
        torch.cuda.set_device(local_rank)
        
        if self.world_size > 1 and self.rank > 0:
            self.loop()
    
    def call(self, method_name: str, *args):
        """调用方法 - 使用 dist.broadcast_object_list 通信"""
        if self.world_size > 1 and self.rank == 0:
            # rank 0 广播方法名和参数
            objects = [[method_name, args]]
            dist.broadcast_object_list(objects, src=0)
        
        if self.world_size > 1 and self.rank > 0:
            # 其他进程接收（在 loop 中处理）
            pass
        
        # 所有进程都执行
        method = getattr(self, method_name, None)
        return method(*args) if method else None
    
    def loop(self):
        """工作进程循环 - 使用 dist.broadcast_object_list"""
        while True:
            objects = [None]
            dist.broadcast_object_list(objects, src=0)
            method_name, args = objects[0]
            
            method = getattr(self, method_name, None)
            if method:
                method(*args)
            if method_name == "exit":
                break


# ============================================================================
# 主要差异总结
# ============================================================================
"""
使用 torchrun 的主要变化：

1. **获取 rank/world_size**:
   - 旧: 手动传入 rank 参数
   - 新: 从环境变量 RANK, WORLD_SIZE 读取，或使用 dist.get_rank()

2. **初始化分布式**:
   - 旧: dist.init_process_group(..., rank=rank, world_size=world_size, ...)
   - 新: dist.init_process_group(backend="nccl")  # torchrun 已设置环境变量

3. **Event 同步**:
   - 旧: multiprocessing.Event（手动创建进程时可传递）
   - 新: 使用 dist.broadcast() 或 dist.broadcast_object_list()（跨进程无需传递对象）

4. **共享内存**:
   - 逻辑不变！仍然可以使用 SharedMemory
   - 但需要确保 shm_name 在 torchrun 启动的所有进程间唯一

5. **启动方式**:
   - 旧: 手动创建 Process
   - 新: torchrun --nproc_per_node=2 script.py
"""
