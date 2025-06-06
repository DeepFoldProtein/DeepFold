import math

import torch
import triton
import triton.language as tl
import triton.testing
from einops import rearrange
from jaxtyping import Bool, Float
from torch.library import triton_op, wrap_triton

from deepfold.ops.autotune import _BWD_B_CONFIGS, _BWD_KV_CONFIGS, _BWD_Q_CONFIGS, _FWD_CONFIGS, autotune


@autotune(configs=_FWD_CONFIGS, key=["H", "DIM", "CLOSEST_N"])
@triton.jit
def _fwd(
    o_ptr,
    stride_oh,
    stride_om,
    stride_on,
    stride_od,
    lse_ptr,
    stride_lseh,
    stride_lsem,
    stride_lsen,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    mask_ptr,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    sm_scale,
    neg_inf,
    N,
    H,
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_h = tl.program_id(2)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0  # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)  # [j,d]

    base_kt_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_kt_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)  # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)  # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd)  # [k,d]

    base_lse_ptr = lse_ptr + (start_h * stride_lseh) + (start_i * stride_lsem)
    lse_ptrs = base_lse_ptr + (j_idxs * stride_lsen)  # [j]

    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)
    mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn)  # [k]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od)  # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

    for start_k in tl.range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N

        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :])  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")  # [k]

        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores, input_precision="ieee")  # [j,k]
        scores *= inv_ln2  # 1.0 / ln(2), [j,k]

        # we want to make scores -inf at mask locations
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        scores = scores - block_max[:, None]  # [j,k]
        exp_scores = tl.math.exp2(scores)  # [j,k]

        summed_exp_scores = tl.sum(exp_scores, 1)  # [j]
        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + summed_exp_scores  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        scores_max = block_max

        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    lse = (scores_max * ln2) + tl.log(sm_denom)

    tl.store(lse_ptrs, lse, mask=mask_j)


@autotune(configs=_BWD_KV_CONFIGS, key=["H", "DIM", "CLOSEST_N"], reset_to_zero=["dk_ptr", "dv_ptr"])
@triton.jit
def _bwd_kv(
    d_ptr,
    stride_dh,
    stride_dm,
    stride_dn,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    l_ptr,
    stride_lh,
    stride_lm,
    stride_ln,
    m_ptr,
    stride_mh,
    stride_mm,
    stride_mn,
    do_ptr,
    stride_doh,
    stride_dom,
    stride_don,
    stride_dod,
    dk_ptr,
    stride_dkh,
    stride_dkm,
    stride_dkn,
    stride_dkd,
    dv_ptr,
    stride_dvh,
    stride_dvm,
    stride_dvn,
    stride_dvd,
    sm_scale,
    neg_inf,
    N,
    H,
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    # program id
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2),

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = 0  # we iterate over j, so each pid starts at 0
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)  # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)  # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)  # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn)  # [d,k]

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln)  # [j]

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs = base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn)  # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)  # [j,d]

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd)  # [k,d]

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd)  # [k,d]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N

    # load k/v once per pid
    vt_block = tl.load(vt_ptrs, mask_k[None, :])  # [d,k]
    kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
    kt_block = kt_block * tl.full([1], value=sm_scale, dtype=input_dtype)  # [k,d]
    m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")  # [k]

    # accumulate over j for dk/dv
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        mask_j = (j_idxs + start_j) < N

        q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)  # [j,k]

        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee")  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores = tl.where(m_block[None, :], neg_inf, scores)

        row_max = tl.load(l_ptrs, mask=mask_j)  # [j]
        sm_value = tl.math.exp2((scores - row_max[:, None]) * inv_ln2)  # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None])  # [j,d]
        dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee")  # [k,d]

        delta = tl.load(d_ptrs, mask_j)  # [j]

        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee")  # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None])  # [j,k]
        dscores = dscores.to(input_dtype)  # [j,k]

        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee")  # [k,d]

        # increment pointers
        q_ptrs += BLOCK_J * stride_qn
        d_ptrs += BLOCK_J * stride_dn
        b_ptrs += BLOCK_J * stride_bm
        l_ptrs += BLOCK_J * stride_ln
        do_ptrs += BLOCK_J * stride_don
        mask_ptrs += BLOCK_J * stride_mn

    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None])
    tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])


@autotune(configs=_BWD_Q_CONFIGS, key=["H", "DIM", "CLOSEST_N"], reset_to_zero=["dq_ptr", "d_ptr"])
@triton.jit
def _bwd_q(
    d_ptr,
    stride_dh,
    stride_dm,
    stride_dn,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    l_ptr,
    stride_lh,
    stride_lm,
    stride_ln,
    mask_ptr,
    stride_maskh,
    stride_maskm,
    stride_maskn,
    o_ptr,
    stride_oh,
    stride_om,
    stride_on,
    stride_od,
    do_ptr,
    stride_doh,
    stride_dom,
    stride_don,
    stride_dod,
    dq_ptr,
    stride_dqh,
    stride_dqm,
    stride_dqn,
    stride_dqd,
    sm_scale,
    neg_inf,
    N,
    H,
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0  # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)  # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd)  # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)  # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn)  # [d,k]

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln)  # [j]

    base_mask_ptr = mask_ptr + (mask_start_h * stride_maskh)
    mask_ptrs = base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn)  # [k]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    base_dq_ptr = dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
    dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs[None, :] * stride_dqd)  # [j,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)  # [j,d]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od)  # [j,d]

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
    sm_denom = tl.load(l_ptrs, mask_j)  # [j]
    do_block = tl.load(do_ptrs, mask_j[:, None])  # [j,d]
    o_block = tl.load(o_ptrs, mask_j[:, None])  # [j,d]

    delta = tl.sum(o_block * do_block, axis=1)  # [j]

    tl.store(d_ptrs, delta.to(input_dtype), mask=mask_j)

    dq_block = tl.zeros([BLOCK_J, DIM], dtype=tl.float32)

    # iterte over k for dq = \sum_{k} ds_{jk} k_{k}
    for start_k in range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N

        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :]).to(tl.float32)  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")  # [k]
        k_block = tl.load(k_ptrs, mask_k[:, None])  # [k,d]
        k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype)  # [j,d]

        scores = tl.dot(q_block, tl.trans(k_block), b_block, input_precision="ieee")  # [j,k]
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        sm_value = tl.math.exp2((scores - sm_denom[:, None]) * inv_ln2)  # [j,k]

        vt_block = tl.load(vt_ptrs, mask_k[None, :])  # [d,k]
        dsm_value = tl.dot(do_block, vt_block, input_precision="ieee")  # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None])  # [j,k]
        dscores = dscores.to(input_dtype)  # [j,k]

        dq_block += tl.dot(dscores, k_block, input_precision="ieee")

        k_ptrs += BLOCK_K * stride_kn
        vt_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])


@autotune(configs=_BWD_B_CONFIGS, key=["H", "DIM", "CLOSEST_N"], reset_to_zero=["db_ptr"])
@triton.jit
def _bwd_b(
    d_ptr,
    stride_dh,
    stride_dm,
    stride_dn,
    q_ptr,
    stride_qh,
    stride_qm,
    stride_qn,
    stride_qd,
    k_ptr,
    stride_kh,
    stride_km,
    stride_kn,
    stride_kd,
    v_ptr,
    stride_vh,
    stride_vm,
    stride_vn,
    stride_vd,
    b_ptr,
    stride_bh,
    stride_bm,
    stride_bn,
    l_ptr,
    stride_lh,
    stride_lm,
    stride_ln,
    m_ptr,
    stride_mh,
    stride_mm,
    stride_mn,
    do_ptr,
    stride_doh,
    stride_dom,
    stride_don,
    stride_dod,
    db_ptr,
    stride_dbh,
    stride_dbm,
    stride_dbn,
    sm_scale,
    neg_inf,
    H,
    M,
    N,
    DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    BLOCK_I: tl.constexpr = 1

    # program id
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2),

    # One mask per batch item, not repeated per head.
    mask_start_h = pid_h // H
    start_h = pid_h
    start_i = 0  # we iterate over j, so each pid starts at 0
    start_j = pid_j * BLOCK_J
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd)  # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd  # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn)  # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd)  # [k,d]

    base_l_ptr = l_ptr + (start_h * stride_lh)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln)  # [j]

    base_mask_ptr = m_ptr + (mask_start_h * stride_mh)
    mask_ptrs = base_mask_ptr + (k_idxs * stride_mn)  # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod)  # [j,d]

    base_db_ptr = db_ptr + (pid_h * stride_dbh)
    db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)

    base_d_ptr = d_ptr + (start_h * stride_dh)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N
    mask_j = j_idxs < N

    db_block = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)

    # loop over i
    for start_i in range(0, M, BLOCK_I):
        start_i = tl.multiple_of(start_i, BLOCK_I)
        q_block = tl.load(q_ptrs, mask_j[:, None], cache_modifier=".cg")  # [j,d]
        q_block *= tl.full([1], value=sm_scale, dtype=input_dtype)
        k_block = tl.load(k_ptrs, mask_k[:, None], cache_modifier=".cg")  # [k,d]

        b_block = tl.load(b_ptrs, mask_j[:, None] & mask_k[None, :], cache_modifier=".cg").to(tl.float32)  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg")  # [k]

        scores = tl.dot(q_block, tl.trans(k_block), b_block, input_precision="ieee")  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores = tl.where(m_block[None, :], neg_inf, scores)

        sm_denom = tl.load(l_ptrs, mask=mask_j, cache_modifier=".cg")  # [j]
        sm_score = tl.math.exp2((scores - sm_denom[:, None]) * inv_ln2)  # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None], cache_modifier=".cg")  # [j,d]
        delta = tl.load(d_ptrs, mask_j, cache_modifier=".cg")  # [j]

        v_block = tl.load(v_ptrs, mask_k[:, None], cache_modifier=".cg")  # [k,d]
        dsm_value = tl.dot(do, tl.trans(v_block), input_precision="ieee")  # [j,k]

        dscores = sm_score * (dsm_value - delta[:, None])  # [j,k]

        db_block += dscores

        # increment pointers
        q_ptrs += stride_qm * BLOCK_I
        k_ptrs += stride_km * BLOCK_I
        v_ptrs += stride_vm * BLOCK_I
        l_ptrs += stride_lm * BLOCK_I
        mask_ptrs += stride_mm * BLOCK_I
        d_ptrs += stride_dm * BLOCK_I
        do_ptrs += stride_dom * BLOCK_I

    tl.store(db_ptrs, db_block.to(input_dtype), mask=mask_j[:, None] & mask_k[None, :])


@triton_op("deepfold::triangle_attention", mutates_args={})
def _triangle_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sm_scale = q.shape[-1] ** -0.5

    bs, h, m, n, dim = q.shape

    # TODO: Should also allow flattening arbitrary batch dims.
    q = rearrange(q, "b h ... -> (b h) ...").contiguous()
    k = rearrange(k, "b h ... -> (b h) ...").contiguous()
    v = rearrange(v, "b h ... -> (b h) ...").contiguous()
    b = rearrange(b, "b h ... -> (b h) ...").contiguous()
    mask = mask.contiguous()

    # e.g. batch x head
    bh = q.shape[0]

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), m, bh)

    o = torch.zeros_like(q)
    l = torch.zeros((bh, m, n), device=q.device, dtype=torch.float32)

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    wrap_triton(_fwd)[grid](
        o,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        l,
        l.stride(0),
        l.stride(1),
        l.stride(2),
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        b,
        b.stride(0),
        b.stride(1),
        b.stride(2),
        mask,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        neg_inf=torch.finfo(q.dtype).min,
        sm_scale=sm_scale,
        N=n,
        H=h,
        DIM=dim,
        CLOSEST_N=CLOSEST_N,
    )

    l = rearrange(l, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return o, l


@triton_op("deepfold::triangle_attention_backward", mutates_args={})
def triangle_attention_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    o: torch.Tensor,
    l: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bs, h, *_ = q.shape

    # TODO: Should also allow flattening arbitrary batch dims.
    q = rearrange(q, "b h ... -> (b h) ...")
    k = rearrange(k, "b h ... -> (b h) ...")
    v = rearrange(v, "b h ... -> (b h) ...")
    b = rearrange(b, "b h ... -> (b h) ...")
    o = rearrange(o, "b h ... -> (b h) ...")
    l = rearrange(l, "b h ... -> (b h) ...")
    do = rearrange(do, "b h ... -> (b h) ...")

    bh, m, n, dim = q.shape
    sm_scale = dim**-0.5

    CLOSEST_N = 2 ** int(math.ceil(math.log2(n)))

    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    db = torch.zeros_like(b)
    dmask = torch.zeros_like(mask)  # Don't need grads, but torch expects a tensor

    d = torch.zeros((bh, m, n), dtype=q.dtype, device=q.device)

    def q_grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), m, bh)

    # NOTE: This also calculates delta for kv/b!
    wrap_triton(_bwd_q)[q_grid](
        d,
        d.stride(0),
        d.stride(1),
        d.stride(2),
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        b,
        b.stride(0),
        b.stride(1),
        b.stride(2),
        l,
        l.stride(0),
        l.stride(1),
        l.stride(2),
        mask,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        o,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        do,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dq,
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h,
        N=n,
        DIM=dim,
        CLOSEST_N=CLOSEST_N,
    )

    # Do the actual backward pass.
    def kv_grid(x):
        return (triton.cdiv(n, x["BLOCK_K"]), m, bh)

    wrap_triton(_bwd_kv)[kv_grid](
        d,
        d.stride(0),
        d.stride(1),
        d.stride(2),
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        b,
        b.stride(0),
        b.stride(1),
        b.stride(2),
        l,
        l.stride(0),
        l.stride(1),
        l.stride(2),
        mask,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        do,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        dk,
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv,
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h,
        N=n,
        DIM=dim,
        CLOSEST_N=CLOSEST_N,
    )

    def b_grid(x):
        return (
            triton.cdiv(n, x["BLOCK_J"]),
            triton.cdiv(n, x["BLOCK_K"]),
            bh,
        )

    wrap_triton(_bwd_b)[b_grid](
        d,
        d.stride(0),
        d.stride(1),
        d.stride(2),
        q,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        b,
        b.stride(0),
        b.stride(1),
        b.stride(2),
        l,
        l.stride(0),
        l.stride(1),
        l.stride(2),
        mask,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        do,
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        db,
        db.stride(0),
        db.stride(1),
        db.stride(2),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h,
        M=m,
        N=n,
        DIM=dim,
        CLOSEST_N=CLOSEST_N,
    )

    dq = rearrange(dq, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dk = rearrange(dk, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dv = rearrange(dv, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    db = rearrange(db, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    return dq, dk, dv, db, dmask


def backwards(ctx, *grad: tuple[Float[torch.Tensor, "B H M N D"],]) -> tuple[
    Float[torch.Tensor, "B H M N D"],  # dq
    Float[torch.Tensor, "B H M N D"],  # dk
    Float[torch.Tensor, "B H M N D"],  # dv
    Float[torch.Tensor, "B H N N"],  # db
    Bool[torch.Tensor, "B M N"],  # dmask
]:
    do = grad[0]
    q, k, v, b, mask, o, l = ctx.saved_tensors
    dq, dk, dv, db, dmask = triangle_attention_bwd(do, q, k, v, b, o, l, mask)

    return dq, dk, dv, db, dmask


def setup_context(ctx, inputs, output) -> None:
    q, k, v, b, mask, *_ = inputs
    o, l = output

    ctx.save_for_backward(q, k, v, b, mask, o, l)


_triangle_attention.register_autograd(backwards, setup_context=setup_context)


def triangle_attention(
    q: Float[torch.Tensor, "B H M N D"],
    k: Float[torch.Tensor, "B H M N D"],
    v: Float[torch.Tensor, "B H M N D"],
    b: Float[torch.Tensor, "B H N N"],
    mask: Bool[torch.Tensor, "B M N"],
) -> Float[torch.Tensor, "B H M N D"]:
    o, _ = _triangle_attention(q, k, v, b, mask)
    return o
