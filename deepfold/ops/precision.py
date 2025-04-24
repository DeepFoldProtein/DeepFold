"""Precision classes and utilities."""

import enum

import torch


@enum.unique
class DotPrecision(enum.Enum):
    """Precision for `dot` operation.

    Naming scheme: {OPERAND_DTYPE}_{ACCUMULATOR_DTYPE}[_{NUM_PASSES}x]
    """

    BF16_F32 = "bf16_f32"

    # GPU only precisions.
    F32_F32 = "f32_f32"  # Full f32 precision (which doesn't use tensor cores).
    TF32_F32 = "tf32_f32"  # Equivalent to `DEFAULT`/`HIGH` on CUDA.
    TF32_F32_3X = "tf32_f32_3x"
    F16_F16 = "f16_f16"
    F16_F32 = "f16_f32"

    @property
    def operand_dtype(self) -> torch.dtype:
        match self:
            case DotPrecision.BF16_F32:
                return torch.bfloat16
            case DotPrecision.F16_F16 | DotPrecision.F16_F32:
                return torch.float16
            case _:
                return torch.float32

    @property
    def accumulator_dtype(self) -> torch.dtype:
        return torch.float16 if (self == DotPrecision.F16_F16) else torch.float32


_TORCH_CUDA_PRECISION_MAP = {
    (torch.float16, "highest"): DotPrecision.F16_F32,
    (torch.bfloat16, "default"): DotPrecision.BF16_F32,
    (torch.float32, "default"): DotPrecision.TF32_F32,
    (torch.float32, "high"): DotPrecision.TF32_F32,
    (torch.float32, "highest"): DotPrecision.F32_F32,
}

_TORCH_CPU_PRECISION_MAP = {
    (torch.float16, "default"): DotPrecision.F16_F32,
    (torch.bfloat16, "default"): DotPrecision.F32_F32,
    (torch.float32, "default"): DotPrecision.F32_F32,
    (torch.float32, "high"): DotPrecision.F32_F32,
    (torch.float32, "highest"): DotPrecision.F32_F32,
}


def _create_torch_precision_map():
    precision_map = {}
    for (dtype, jax_precision), dot_precision in _TORCH_CUDA_PRECISION_MAP.items():
        precision_map[("cuda", dtype, jax_precision)] = dot_precision
    for (dtype, jax_precision), dot_precision in _TORCH_CPU_PRECISION_MAP.items():
        precision_map[("cpu", dtype, jax_precision)] = dot_precision
    return precision_map


_TORCH_PRECISION_MAP = _create_torch_precision_map()


def get_equivalent_dot_precision(
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    matmul_precision: str,
    backend: str,
) -> DotPrecision:
    """Returns `DotPrecision` replicating default XLA behaviour."""
    if a_dtype != b_dtype:
        raise ValueError("Cannot infer precision if operand types differ.")

    if (matmul_precision != "default") and (a_dtype != torch.float32):
        raise ValueError("Precision values other than `DEFAULT` only have an effect if the operand type is `float32`.")
    return _TORCH_PRECISION_MAP[(backend, a_dtype, matmul_precision)]
