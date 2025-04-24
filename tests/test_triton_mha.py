import math
import random
import unittest

import torch

torch.set_default_device("cuda:0")


from deepfold.modules.attention import _attention_eager as torch_mha
from deepfold.ops.triton_mha import mha as triton_mha


def random_attention_inputs_v1(
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    c_hidden: int,
    # inf: float = 1e9,
    dtype: torch.dtype = torch.float32,
    requires_grad: bool = False,
):
    q = torch.randn([batch_size, num_tokens, num_heads, c_hidden], dtype=dtype, requires_grad=requires_grad)
    k = torch.randn([batch_size, num_tokens, num_heads, c_hidden], dtype=dtype, requires_grad=requires_grad)
    v = torch.randn([batch_size, num_tokens, num_heads, c_hidden], dtype=dtype, requires_grad=requires_grad)

    mask = torch.randint(0, 2, [batch_size, num_heads, num_tokens, num_tokens], dtype=torch.bool, requires_grad=False)
    bias = torch.randn([batch_size, num_heads, num_tokens, num_tokens], dtype=dtype, requires_grad=requires_grad)

    return (q, k, v, mask, bias)


def random_attention_inputs_v2(B, Tq, Tk, Hq, Hk, D, dtype: torch.dtype):
    q = torch.randn(B, Tq, Hq, D, dtype=dtype)
    k = torch.randn(B, Tk, Hk, D, dtype=dtype)
    v = torch.randn(B, Tk, Hk, D, dtype=dtype)

    # full broadcast bias & mask
    bias = torch.randn(B, Hq, Tq, Tk, dtype=dtype)
    mask = torch.randint(0, 2, (B, Hq, Tq, Tk), dtype=torch.bool)

    return q, k, v, mask, bias


class TestTritonAttentionKernel(unittest.TestCase):
    def compare_forward(self):
        for _ in range(16):
            batch_size = random.randint(1, 16)
            num_tokens = random.randint(64, 2048)
            num_heads = random.choice([4, 8])
            c_hidden = random.choice([16, 32])

            q, k, v, mask, bias = random_attention_inputs_v1(batch_size, num_tokens, num_heads, c_hidden)
            real_out = torch_mha(q, k, v, mask, bias, 1e8)
            triton_out = triton_mha(q, k, v, mask, bias)

            is_allclose = torch.allclose(triton_out, real_out, rtol=1e-05, atol=1e-02, equal_nan=False)
            self.assertTrue(is_allclose)


def main_v1():
    for i in range(16):
        batch_size = random.randint(1, 16)
        num_tokens = random.randint(64, 2048)
        num_heads = random.choice([4, 8])
        c_hidden = random.choice([16, 32])
        shape = (batch_size, num_tokens, num_heads, c_hidden)
        num_elem = math.prod(shape)
        print(f"iter={i:02d} shape={shape} {num_elem}", end="")

        q, k, v, mask, bias = random_attention_inputs_v1(batch_size, num_tokens, num_heads, c_hidden, dtype=torch.bfloat16)
        bias = None
        scaling = 1.0 / math.sqrt(q.size(-1))
        real_out = torch_mha(q.transpose(-2, -3), k.transpose(-2, -3), v.transpose(-2, -3), mask.float(), bias, 1e8).transpose(-2, -3).cpu()
        triton_out = triton_mha(q, k, v, mask, bias, scaling, 1e8).cpu()

        rtol = 1e-05  # 1e-05
        atol = 1e-02  # 1e-08

        diff = torch.abs(real_out - triton_out)

        cond = torch.less_equal(diff, atol + rtol * torch.abs(real_out))
        if not torch.all(cond):
            std, mean = torch.std_mean(diff[~cond], correction=0)
            std = std.item()
            mean = mean.item()
            print(f" errors={(~cond).sum()/num_elem*100:05.4f}% std={std:010.8e} mean={mean:010.8e}")
            print(diff[~cond].flatten()[:36])
        else:
            print()
        del q, k, v, mask, bias, real_out, triton_out


def main_v2():
    B, Tq, Tk, Hq, Hk, D = 2, 128, 256, 4, 4, 64
    q, k, v, mask, bias = random_attention_inputs_v2(B, Tq, Tk, Hq, Hk, D, torch.float32)

    inf = 1e08
    logits_scale = 1 / math.sqrt(D)
    out_triton = triton_mha(q, k, v, mask, bias, logits_scale, inf)

    # PyTorch reference
    q_ = q.permute(0, 2, 1, 3)
    k_ = k.permute(0, 2, 1, 3)
    v_ = v.permute(0, 2, 1, 3)
    bias_ = bias
    mask_ = -inf * ~mask + bias_

    ref = torch.nn.functional.scaled_dot_product_attention(
        q_,
        k_,
        v_,
        attn_mask=mask_,  # PyTorch uses True=forbid
        dropout_p=0.0,
        scale=logits_scale,
        is_causal=False,
    )  # [B*Hq, Tq, D]
    out_torch = ref.permute(0, 2, 1, 3)

    rtol = 1e-03  # 1e-05
    atol = 1e-03  # 1e-08

    err = torch.abs(out_torch - out_triton)
    pos = torch.less_equal(err, atol + rtol * out_torch.abs())
    print("max_abs_err=", err.max().item())
    print("num_err=", (~pos).sum().item())
    print(err[~pos])


if __name__ == "__main__":
    # main_v1()
    unittest.main()
