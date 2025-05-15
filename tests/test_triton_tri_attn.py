import unittest
from itertools import product

import torch

from deepfold.ops.alter_mha import attention_reference
from deepfold.ops.triton_mha import triangle_attention
from deepfold.ops.utils import disable_tf32, enable_tf32, sample_tensors
from tests.utils import clone_and_clear_grad, compare_values, set_seed

set_seed(1398)

dtype_eps = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-4,
}


class TestTritonAttention(unittest.TestCase):
    def test_values(self):
        for dtype, mask, bs, std, (n, h, d) in product(
            [torch.float32, torch.bfloat16, torch.float16],
            [True, False],
            [1, 2],
            [1.0, 2.0],
            [
                (16, 1, 16),
                (32, 1, 32),
                (64, 1, 64),
                (16, 4, 128),
                *[(n, 4, 32) for n in range(16, 256, 4)],
                # (191, 4, 32),
            ],
        ):
            device = torch.device("cuda")
            q, k, v, b, m = sample_tensors(n, d, h, use_mask=mask, device=device, dtype=torch.float32, batch=bs, std=std, m=48)
            torch.cuda.synchronize()

            o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
            o_ref.sum().backward()

            dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

            o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
            o_kernel.sum().backward()
            dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

            o_pt = enable_tf32(attention_reference)(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
            o_pt.sum().backward()
            dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

            compare_values(o_kernel, o_pt, o_ref, f"o failed ({dtype})", eps=dtype_eps[dtype])
            compare_values(dq_kernel, dq_pt, dq_ref, f"dq failed ({dtype})", eps=dtype_eps[dtype])
            compare_values(dk_kernel, dk_pt, dk_ref, f"dk failed ({dtype})", eps=dtype_eps[dtype])
            compare_values(dv_kernel, dv_pt, dv_ref, f"dv failed ({dtype})", eps=dtype_eps[dtype])
            compare_values(db_kernel, db_pt, db_ref, f"db failed ({dtype})", eps=dtype_eps[dtype])
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
