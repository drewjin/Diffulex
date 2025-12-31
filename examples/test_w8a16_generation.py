#!/usr/bin/env python3
"""测试 W8A16 Linear 量化策略的文本生成"""
import os
import sys
import time
from pathlib import Path

# 确保从当前仓库导入
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer
from diffulex import Diffulex, SamplingParams


def test_generation(llm, tokenizer, test_name: str, prompts: list[str]):
    """运行文本生成测试"""
    print("\n" + "=" * 70)
    print(f"测试: {test_name}")
    print("=" * 70)
    
    sampling_params = SamplingParams(temperature=0.7, max_tokens=30)
    
    # 添加 BOS token（如果需要）
    prompts_with_bos = []
    for p in prompts:
        if tokenizer.bos_token and not p.startswith(tokenizer.bos_token):
            prompts_with_bos.append(tokenizer.bos_token + p)
        else:
            prompts_with_bos.append(p)
    
    print(f"输入 prompts ({len(prompts_with_bos)} 个):")
    for i, p in enumerate(prompts_with_bos, 1):
        print(f"  {i}. {p[:60]}...")
    
    print(f"\n开始生成...")
    start_time = time.time()
    
    try:
        outputs = llm.generate(prompts_with_bos, sampling_params)
        end_time = time.time()
        
        total_time = end_time - start_time
        total_tokens = sum(len(o.get('token_ids', [])) for o in outputs)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        print(f"\n✓ 生成成功!")
        print(f"  - 总时间: {total_time:.2f} 秒")
        print(f"  - 总 token 数: {total_tokens}")
        print(f"  - 平均 TPS: {avg_tps:.2f} tok/s")
        
        print(f"\n生成结果:")
        for i, output in enumerate(outputs, 1):
            generated_text = output.get('text', '')
            token_ids = output.get('token_ids', [])
            print(f"\n  [{i}] 输入: {prompts[i-1][:50]}...")
            print(f"       输出: {generated_text[:150]}...")
            print(f"       Token数: {len(token_ids)}")
        
        return True
    except Exception as e:
        print(f"\n✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # 检查模型路径
    model_path = os.getenv("DIFFULEX_TEST_MODEL", "/data1/ckpts/Dream-org/Dream-v0-Base-7B")
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请设置环境变量 DIFFULEX_TEST_MODEL 指向有效的模型路径")
        return
    
    print("=" * 70)
    print("Diffulex W8A16 Linear 量化文本生成测试")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    
    # 测试 prompts
    test_prompts = [
        "The capital of France is",
        "Python is a programming language",
    ]
    
    # 加载 tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print(f"✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        return
    
    # 测试: W8A16 路径 (int8 weight + bf16 activation)
    print("\n" + "=" * 70)
    print("测试: W8A16 Linear 量化 (int8 weight + bf16 activation)")
    print("=" * 70)
    
    try:
        llm_w8a16 = Diffulex(
            model_path,
            lora_path=os.getenv("DIFFULEX_TEST_LORA", ""),
            use_lora=bool(os.getenv("DIFFULEX_TEST_LORA", "")),
            model_name="dream",
            enforce_eager=True,
            data_parallel_size=1,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.3,
            max_num_batched_tokens=1024,
            max_num_seqs=4,
            max_model_len=1024,
            kv_cache_dtype="bf16",
            kv_cache_layout="unified",
            decoding_strategy="d2f",
            # W8A16 配置
            linear_attn_weight_dtype="int8",
            linear_mlp_weight_dtype="int8",
            linear_attn_act_dtype="bf16",
            linear_mlp_act_dtype="bf16",
        )
        print("✓ W8A16 模型初始化成功")
        
        test_generation(llm_w8a16, tokenizer, "W8A16 Linear 量化", test_prompts)
        
        # 清理
        llm_w8a16.exit()
        del llm_w8a16
        import torch
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ W8A16 路径测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()

