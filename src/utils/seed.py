"""Deterministic seeding helpers.

Note: even with seeds set, runs are not bit-exact across GPU types or CUDA
versions due to nondeterministic reductions in fused kernels. They are
reproducible on the same hardware + driver + cudnn version.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic_cudnn: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic_cudnn:
        # Slower; usually only enable when chasing a specific reproducibility issue.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
