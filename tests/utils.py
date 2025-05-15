import os
import random

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clone_and_clear_grad(*tensors):
    """
    Clone gradients of tensors and clear them.
    Returns a tuple of cloned gradients.
    """
    grads = tuple(t.grad.clone() if t.grad is not None else None for t in tensors)
    for t in tensors:
        t.grad = None
    return grads


def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)
    # Numpy random seed
    np.random.seed(0)
    # PyTorch random seed
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

        # PyTorch backend settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def compare_values(tri, pt, ref, msg="", eps=1e-4):
    a = (tri.float() - ref.float()).abs().max().item()
    a_std = (tri.float() - ref.float()).abs().std().item()
    b = (pt.float() - ref.float()).abs().max().item()

    # This factor of 3 is pretty arbitrary.
    assert a <= 3 * (b + eps), f"{msg} value mismatch, tri: <{a:.3e} (±{a_std:.3f}), pt: <{b:.3e}"
