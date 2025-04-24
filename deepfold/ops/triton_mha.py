"""Triton FlashAttention implementation."""

import jaxtyping
import torch
import triton
import triton.language as tl
from beartype import beartype as typechecker
from jaxtyping import Bool, Float


@triton.jit
def dot_fn(a: tl.tensor, b: tl.tensor, in_dtype: tl.dtype, out_dtype: tl.dtype):
    if in_dtype == tl.float32:
        tl.static_assert(a.dtype == tl.float32)
        tl.static_assert(b.dtype == tl.float32)
    else:
        tl.static_assert(a.dtype.is_standard_floating())
        tl.static_assert(b.dtype.is_standard_floating())
    a = a.to(in_dtype)
    b = b.to(in_dtype)
    return tl.dot(a, b, input_precision="tf32x3", out_dtype=out_dtype)


@triton.jit
def _fwd_kernel_inner(
    start_loop,
    end_loop,
    q,
    k_block_ptr,
    v_block_ptr,
    mask_block_ptr,
    bias_block_ptr,
    seq_len_k,
    acc,
    m_i,
    l_i,
    mask_value: tl.constexpr,
    mask_advance: tl.constexpr,
    bias_advance: tl.constexpr,
    mask_start_dim: tl.constexpr,
    bias_start_dim: tl.constexpr,
    block_k: tl.constexpr,
):
    """Triton MHA forward kernel's inner loop."""
    for start_k in range(start_loop, end_loop, block_k):
        start_k = tl.multiple_of(start_k, block_k)
        span_k = start_k + tl.arange(0, block_k)

        # Load k as before (axis 0=head_dim, axis1=seq)
        k = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        # Load v guarding sequence axis (first axis)
        v = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Prevent dot accumulating into the bias tensor. It appears that Triton
        # doesn't pipeline the bias load as it does the `k` load, so the bias load
        # blocks the matmul if the add is merged.

        # Load bias
        bias = tl.load(
            bias_block_ptr,
            boundary_check=(0,) if bias_start_dim else (0, 1),
            padding_option="zero",
        )

        # Load q guarding sequence axis
        qk = dot_fn(q.to(k.dtype), k, k.dtype, tl.float32)  # [block_q, block_k]

        # Bias add
        # qk = qk.to(tl.uint32, bitcast=True) & 0xFFFFFFFF
        # qk = qk.to(tl.float32, bitcast=True)
        qk += bias

        # Mask apply
        # mask_value = float(torch.finfo(torch.float32).min)
        mask = tl.load(
            mask_block_ptr,
            boundary_check=(0,) if mask_start_dim else (0, 1),
            padding_option="zero",
        ).to(tl.int1)
        qk = tl.where(mask, qk, mask_value)

        # Guard kv length
        # if seq_len_k % block_k != 0:  # block_q
        qk = tl.where((span_k < seq_len_k)[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))  # Shape [block_q].
        p = tl.exp(qk - m_ij[:, None])  # Shape [block_q, block_k].
        alpha = tl.exp(m_i - m_ij)
        m_i = m_ij
        acc *= alpha[:, None]
        l_i *= alpha
        l_i += tl.sum(p, axis=1)

        # Add the new block of attention weights.
        acc += dot_fn(p.to(v.dtype), v, v.dtype, tl.float32)

        k_block_ptr = tl.advance(k_block_ptr, (0, block_k))
        v_block_ptr = tl.advance(v_block_ptr, (block_k, 0))
        mask_block_ptr = tl.advance(mask_block_ptr, mask_advance)
        bias_block_ptr = tl.advance(bias_block_ptr, bias_advance)

    return (
        k_block_ptr,
        v_block_ptr,
        mask_block_ptr,
        bias_block_ptr,
        acc,
        m_i,
        l_i,
    )


configs = [
    triton.Config(
        {"block_q": block_q, "block_k": block_k},
        num_stages=s,
        num_warps=w,
    )
    for block_q in [64, 128]
    for block_k in [32, 64]
    for s in [3, 4, 7]
    for w in [4, 8]
]


def keep(conf: triton.Config) -> bool:
    block_q: int = conf.kwargs["block_q"]
    block_k: int = conf.kwargs["block_k"]
    if block_q * block_k < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# Based on Algorithm 1 of https://arxiv.org/abs/2205.14135.
@triton.autotune(list(filter(keep, configs)), key=["head_dim"])
@triton.jit(debug=True)
def _fwd_kernel(
    # Input arrays.
    q_ptr,
    k_ptr,
    v_ptr,
    # Output array.
    o_ptr,
    # Bias arrays.
    mask_ptr,
    bias_ptr,
    # Scalar inputs.
    q_offset,
    k_offset,
    v_offset,
    q_stride_b,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_b,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    v_stride_b,
    v_stride_s,
    v_stride_h,
    v_stride_d,
    o_stride_b,
    o_stride_s,
    o_stride_h,
    o_stride_d,
    mask_stride_b,
    mask_stride_h,
    mask_stride_sq,
    mask_stride_sk,
    bias_stride_b,
    bias_stride_h,
    bias_stride_sq,
    bias_stride_sk,
    num_heads_q,
    num_heads_k,
    seq_len_q,
    seq_len_k,
    # Compile-time constants.
    sm_scale: tl.constexpr,
    head_dim: tl.constexpr,
    mask_value: tl.constexpr,
    mask_bcast_sq: tl.constexpr,
    bias_bcast_sq: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
):
    """Triton MHA forward kernel."""
    block_d: tl.constexpr = triton.next_power_of_2(head_dim.value)

    # Each thread block processes one batch element (b) and one head (h).
    start_q = tl.program_id(1) * block_q
    off_h = tl.program_id(0)  # int in [0, num_heads_o).
    off_b = tl.program_id(2)  # int in [0, batch_size)

    off_h_k = off_h // (num_heads_q // num_heads_k)

    q_ptr += off_h * q_stride_h + off_b * q_stride_b + q_offset
    k_ptr += off_h_k * k_stride_h + off_b * k_stride_b + k_offset
    v_ptr += off_h_k * v_stride_h + off_b * v_stride_b + v_offset
    o_ptr += off_h * o_stride_h + off_b * o_stride_b
    bias_ptr += off_b * bias_stride_b + off_h * bias_stride_h
    mask_ptr += off_b * mask_stride_b + off_h * mask_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr,
        shape=(seq_len_q, head_dim),
        strides=(q_stride_s, q_stride_d),
        offsets=(start_q, 0),
        block_shape=(block_q, block_d),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr,
        shape=(head_dim, seq_len_k),
        strides=(k_stride_d, k_stride_s),
        offsets=(0, 0),
        block_shape=(block_d, block_k),
        order=(0, 1),
    )
    v_block_ptr = tl.make_block_ptr(
        v_ptr,
        shape=(seq_len_k, head_dim),
        strides=(v_stride_s, v_stride_d),
        offsets=(0, 0),
        block_shape=(block_k, block_d),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr,
        shape=(seq_len_q, head_dim),
        strides=(o_stride_s, o_stride_d),
        offsets=(start_q, 0),
        block_shape=(block_q, block_d),
        order=(1, 0),
    )

    # If broadcasting in a given dim, use a 1D block (observed to be faster).
    bias_start_dim: tl.constexpr = 1 if bias_bcast_sq else 0
    bias_block_ptr = tl.make_block_ptr(
        bias_ptr,
        shape=(seq_len_q, seq_len_k)[bias_start_dim:],
        strides=(bias_stride_sq, bias_stride_sk)[bias_start_dim:],
        offsets=(start_q, 0)[bias_start_dim:],
        block_shape=(block_q, block_k)[bias_start_dim:],
        order=(1, 0)[bias_start_dim:],
    )
    bias_advance: tl.constexpr = (0, block_k)[bias_start_dim:]
    # bias_advance = (block_k,) if bias_start_dim == 1 else (0, block_k)

    mask_start_dim: tl.constexpr = 1 if mask_bcast_sq else 0
    mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(seq_len_q, seq_len_k)[mask_start_dim:],
        strides=(mask_stride_sq, mask_stride_sk)[mask_start_dim:],
        offsets=(start_q, 0)[mask_start_dim:],
        block_shape=(block_q, block_k)[mask_start_dim:],
        order=(1, 0)[mask_start_dim:],
    )
    mask_advance: tl.constexpr = (0, block_k)[mask_start_dim:]
    # mask_advance = (block_k,) if mask_start_dim == 1 else (0, block_k)

    # Each thread block processes a block of block_q queries.
    # span_q = start_q + tl.arange(0, block_q)

    # m_i and l_i (see FlashAttention paper) are updated during the k,v loop.
    m_i = tl.full([block_q], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([block_q], dtype=tl.float32)
    # acc is the buffer where we accumulate the output on SRAM.
    acc = tl.zeros([block_q, block_d], dtype=tl.float32)

    # Load q: it will stay in smem throughout. Indices form a matrix because we
    # read, compute, and write all in 2d chunks. 1 element ~= 1 CUDA thread index.
    # use_mask_q = seq_len_q % block_q != 0
    # q_boundary_check0 = (0,) if use_mask_q else ()
    # q_boundary_check1 = (1,) if head_dim != block_d else ()
    # q_boundary_check = q_boundary_check0 + q_boundary_check1
    # q_padding_option = "zero" if len(q_boundary_check.value) else ""
    # q = tl.load(q_block_ptr, boundary_check=q_boundary_check, padding_option=q_padding_option)
    q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q *= sm_scale

    # In FlashAttention algorithm 1 there are 2 loops: slow over tiles of kv (size
    # (Bc == block_k here), and fast over blocks of q (size Br == block_q here).
    # Here we only loop over blocks of kv to process entire seq_len, the loop over
    # blocks of q is carried out by the grid.
    start_loop = 0
    end_loop = seq_len_k

    (_, _, _, _, acc, _, l_i) = _fwd_kernel_inner(
        start_loop,
        end_loop,
        q,
        k_block_ptr,
        v_block_ptr,
        mask_block_ptr,
        bias_block_ptr,
        seq_len_k,
        acc,
        m_i,
        l_i,
        mask_value,
        mask_advance,
        bias_advance,
        mask_start_dim,
        bias_start_dim,
        block_k,
    )

    # It is possible that every value in a row was masked to f32 min or that the
    # main loop has been completely optimised out, and that `l_i` is `0` for that
    # row. Add epsilon value to avoid NaNs from `0 / 0`.
    l_i += float(torch.finfo(torch.float32).tiny)

    acc /= l_i[:, None]

    # Write output to DRAM.

    acc = acc.to(o_ptr.dtype.element_ty)
    tl.store(o_block_ptr, acc, boundary_check=(0, 1))


@jaxtyping.jaxtyped(typechecker=typechecker)
def _fwd(
    q: Float[torch.Tensor, "*B T H D"],
    k: Float[torch.Tensor, "*B t H D"],
    v: Float[torch.Tensor, "*B t H D"],
    mask: Bool[torch.Tensor, "*#B #H #T #t"],
    bias: Float[torch.Tensor, "*#B #H #T #t"] | None = None,
    *,
    logits_scale: float,
    inf: float = float(torch.finfo(torch.float32).max),
) -> Float[torch.Tensor, "*B T H D"]:
    # print()
    # print(
    #     "triton_mha_fwd:",
    #     "q:",
    #     tuple(q.shape),
    #     q.stride(),
    #     "k:",
    #     tuple(k.shape),
    #     k.stride(),
    #     "v:",
    #     tuple(v.shape),
    #     v.stride(),
    #     "mask:",
    #     tuple(mask.shape),
    #     mask.stride(),
    #     "bias:",
    #     tuple(bias.shape) if bias is not None else None,
    #     bias.stride() if bias is not None else None,
    # )

    """Forward pass of Triton FlashAttention."""
    orig_q_shape = q.shape
    if q.stride(-1) != 1:
        q = q.contiguous()
    q = q.flatten(0, -4)
    o = torch.empty_like(q)
    batch_size, seq_len_q, num_heads_q, head_dim = q.shape
    *_, seq_len_k, num_heads_kv, _ = k.shape
    kv_shape = (batch_size, seq_len_k, num_heads_kv, head_dim)
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    k = k.flatten(0, -4).broadcast_to(kv_shape)
    v = v.flatten(0, -4).broadcast_to(kv_shape)

    shape = orig_q_shape[:-3] + (num_heads_q, seq_len_q, seq_len_k)
    mask = mask.broadcast_to(shape).flatten(0, -4)

    if bias is None:
        bias = torch.tensor(0, device=q.device, dtype=q.dtype)
    shape = orig_q_shape[:-3] + (num_heads_q, seq_len_q, seq_len_k)
    bias = bias.broadcast_to(shape).flatten(0, -4)

    def grid(META: dict):
        block_q = META["block_q"]
        return (num_heads_q, triton.cdiv(seq_len_q, block_q), batch_size)

    # print(
    #     "  ->",
    #     "q:",
    #     tuple(q.shape),
    #     q.stride(),
    #     "k:",
    #     tuple(k.shape),
    #     k.stride(),
    #     "v:",
    #     tuple(v.shape),
    #     v.stride(),
    #     "b:",
    #     tuple(bias.shape),
    #     bias.stride(),
    #     "m:",
    #     tuple(mask.shape),
    #     mask.stride(),
    #     "o:",
    #     tuple(o.shape),
    #     o.stride(),
    # )

    assert num_heads_q % num_heads_kv == 0
    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        mask,
        bias,
        0,  # q.offset
        0,  # k.offset
        0,  # v.offset
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *o.stride(),
        *mask.stride(),
        *bias.stride(),
        num_heads_q,
        num_heads_kv,
        seq_len_q,
        seq_len_k,
        sm_scale=logits_scale,
        head_dim=head_dim,
        mask_value=-inf,
        mask_bcast_sq=int(mask.stride(-2) == 0 and mask.stride(-1) != 0),
        bias_bcast_sq=int(bias.stride(-2) == 0 and bias.stride(-1) != 0),
    )

    return o.reshape(orig_q_shape)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, mask, bias, logits_scale, inf):
        return _fwd(
            q=query,
            k=key,
            v=value,
            mask=mask,
            bias=bias,
            logits_scale=logits_scale,
            inf=inf,
        )


mha = TritonFlashAttention.apply
