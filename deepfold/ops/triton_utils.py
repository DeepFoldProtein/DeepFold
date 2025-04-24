"""Triton utilities."""

from collections.abc import Callable, Mapping

import numpy as np
import torch
import triton
import triton.language as tl

import deepfold.ops.precision as precision_lib

_TORCH_TO_TL_DTYPES: Mapping[torch.dtype, tl.dtype] = {
    torch.bool: tl.int1,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
    torch.uint16: tl.uint16,
    torch.uint32: tl.uint32,
    torch.uint64: tl.uint64,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
}


def torch_to_tl_dtype(torch_dtype: torch.dtype) -> tl.dtype:
    return _TORCH_TO_TL_DTYPES[torch_dtype]


def get_tl_dot_fn(precision: precision_lib.DotPrecision) -> Callable[..., tl.tensor]:
    """Returns a tl `dot` implementation with the specified precision.

    Args:
      precision: The `dot` precision.
    """
    if not is_precision_supported(precision):
        raise ValueError(f"Unsupported dot precision: {precision}")

    if precision == precision_lib.DotPrecision.TF32_F32_3X:
        return _dot_tf32_f32_3x

    in_dtype = torch_to_tl_dtype(precision.operand_dtype)
    out_dtype = torch_to_tl_dtype(precision.accumulator_dtype)
    allow_tf32 = precision == precision_lib.DotPrecision.TF32_F32

    @tl.core.extern
    # @triton.jit
    def _dot_fn(
        a: tl.core.tensor,
        b: tl.core.tensor,
        *,
        trans_a: bool = False,
        trans_b: bool = False,
        _builder,
    ):
        if in_dtype == tl.float32:
            tl.static_assert(a.dtype == tl.float32, _builder=_builder)
            tl.static_assert(b.dtype == tl.float32, _builder=_builder)
        else:
            tl.static_assert(a.dtype.is_standard_floating(), _builder=_builder)
            tl.static_assert(b.dtype.is_standard_floating(), _builder=_builder)
        a = a.to(in_dtype, _builder=_builder)
        b = b.to(in_dtype, _builder=_builder)
        a = tl.trans(a, _builder=_builder) if trans_a else a
        b = tl.trans(b, _builder=_builder) if trans_b else b
        return tl.dot(a, b, allow_tf32=allow_tf32, out_dtype=out_dtype, _builder=_builder)

    return _dot_fn


def is_precision_supported(precision: precision_lib.DotPrecision) -> bool:
    return precision in {
        precision_lib.DotPrecision.F32_F32,
        precision_lib.DotPrecision.TF32_F32,
        precision_lib.DotPrecision.F16_F32,
        precision_lib.DotPrecision.BF16_F32,
        precision_lib.DotPrecision.TF32_F32_3X,
    }


@triton.jit
def _dot_tf32_f32_3x(a, b, trans_a=False, trans_b=False):
    """Perform the 3-pass tf32 dot function."""
    tl.static_assert(a.dtype == tl.float32)
    tl.static_assert(b.dtype == tl.float32)
    a_ = (a.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
    b_ = (b.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
    a_err = a - a_
    b_err = b - b_
    if trans_a:
        a_ = tl.trans(a_)
        a_err = tl.trans(a_err)
    if trans_b:
        b_ = tl.trans(b_)
        b_err = tl.trans(b_err)
    # Add smallest terms first for better accuracy.
    return tl.dot(a_, b_, out_dtype=tl.float32) + (tl.dot(a_, b_err, out_dtype=tl.float32) + tl.dot(a_err, b_, out_dtype=tl.float32))


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    size = np.prod(shape)
    strides = []
    for s in shape:
        size = size // s
        strides.append(int(size))
    return tuple(strides)
