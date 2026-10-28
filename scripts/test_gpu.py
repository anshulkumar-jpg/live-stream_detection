#!/usr/bin/env python3
"""Verify CUDA from inside the worker container; exits non-zero with a clear message when unavailable."""
import sys

try:
    import torch
except ImportError:
    sys.exit("FAIL: torch not installed (build the image with WITH_GPU=1)")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False. Check: nvidia-smi on the host, NVIDIA Container Toolkit, "
             "compose GPU reservation / --gpus, driver version vs torch CUDA " + str(torch.version.cuda))
n = torch.cuda.device_count()
print(f"CUDA OK: torch {torch.__version__}, cuda {torch.version.cuda}, {n} visible device(s) in THIS container")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB (container index {i}; host index may differ)")
x = torch.randn(1024, 1024, device="cuda")
print("matmul ok:", float((x @ x).sum().isfinite()))
