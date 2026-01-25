"""
AWQ W4A16 Linear quantization strategy (vLLM standard format).

- Weight format: vLLM AWQ (packed int32 qweight/qzeros + fp16 scales)
- Activation: bf16 (no activation quantization)
- Forward: vLLM custom op `awq_gemm` (with the same heuristic as vLLM)

No TileLang dependency.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from diffulex.utils.quantization.registry import register_linear_strategy
from diffulex.utils.quantization.strategy import LinearQuantizationStrategy

try:
    from vllm import _custom_ops as ops  # type: ignore
except Exception:  # pragma: no cover
    ops = None  # type: ignore


@register_linear_strategy(weight_dtype="awq", act_dtype="bf16")
def _build_linear_awq_w4a16() -> LinearQuantizationStrategy:
    return LinearAWQW4A16Strategy()


class LinearAWQW4A16Strategy(LinearQuantizationStrategy):
    def __init__(self) -> None:
        super().__init__()
        self._ops_available: bool = bool(ops is not None and hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "awq_gemm"))

    @property
    def name(self) -> str:
        return "linear_awq_w4a16"

    @property
    def linear_weight_format(self) -> str:
        return "awq"

    @property
    def linear_act_format(self) -> str:
        return "bf16"

    def get_storage_dtype(self) -> tuple[torch.dtype, int]:
        # vLLM AWQ stores packed weights in int32.
        return torch.int32, 4

    def get_scale_shape(self, original_shape: tuple[int, ...], **kwargs) -> tuple[int, ...]:
        # vLLM AWQ scales: [K/group, N], where Linear weight is (N, K).
        if len(original_shape) != 2:
            raise ValueError(f"Expected 2D weight shape, got {original_shape}")
        out_features, in_features = original_shape
        group_size = int(kwargs.get("group_size", 128))
        group_size = in_features if group_size == -1 else group_size
        if group_size <= 0 or in_features % group_size != 0:
            raise ValueError(f"Invalid group_size={group_size} for in_features={in_features}")
        num_groups = in_features // group_size
        return (num_groups, out_features)

    def quantize(self, tensor: torch.Tensor, **kwargs) -> tuple[torch.Tensor, Any]:
        # Offline AWQ is handled by `diffulex.utils.quantization.quantize_model`.
        return tensor, {}

    def dequantize(self, quantized: torch.Tensor, scale_or_metadata: Any, **kwargs) -> torch.Tensor:
        if quantized.is_floating_point():
            return quantized
        raise NotImplementedError(
            "AWQ dequantize is not implemented in Diffulex. "
            "Use vLLM kernels via linear_forward."
        )

    def linear_forward(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        *,
        quant_kind: str,
        awq_qweight: Optional[torch.Tensor] = None,
        awq_qzeros: Optional[torch.Tensor] = None,
        awq_scales: Optional[torch.Tensor] = None,
        pack_factor: int = 8,
        out_features: Optional[int] = None,
        in_features: Optional[int] = None,
        group_size: int = 128,
    ) -> torch.Tensor:
        _ = quant_kind, weight, pack_factor, in_features, group_size
        if not self._ops_available:
            raise RuntimeError(
                "vLLM is required for AWQ W4A16 (missing `vllm._custom_ops`). "
                "Please install/build vLLM with CUDA ops."
            )
        qweight = awq_qweight
        qzeros = awq_qzeros
        scales = awq_scales
        if qweight is None or qzeros is None or scales is None:
            if weight is None:
                raise RuntimeError("AWQ offline weights missing packed tensors and bf16 weight is not present.")
            return F.linear(x, weight, bias)

        # vLLM AWQ kernels expect FP16 activations.
        x_in = x if x.dtype == torch.float16 else x.to(dtype=torch.float16)

        # Use known out_features if provided (avoid per-call inference).
        n = int(out_features) if out_features is not None else int(scales.shape[1])
        out_shape = x.shape[:-1] + (n,)
        reshaped_x = x_in.reshape(-1, x_in.shape[-1])

        # Always use awq_gemm to avoid large temporary dequantized weight allocations.
        # vLLM API: awq_gemm(input, qweight, qzeros, scales, split_k_iters)
        split_k_iters = 1
        if reshaped_x.is_contiguous() and qweight.is_contiguous() and qzeros.is_contiguous() and scales.is_contiguous():
            out = torch.ops._C.awq_gemm(reshaped_x, qweight, qzeros, scales, split_k_iters)
        else:
            out = ops.awq_gemm(reshaped_x, qweight, qzeros, scales, split_k_iters)

        if bias is not None:
            out.add_(bias.to(dtype=out.dtype))
        out = out.reshape(out_shape)
        return out.to(dtype=x.dtype) if out.dtype != x.dtype else out

