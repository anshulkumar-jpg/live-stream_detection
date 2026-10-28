#!/usr/bin/env python3
"""Report what this host/container can actually do: Python, deps, FFmpeg (NVDEC/NVENC *tested*), CUDA, models."""
import importlib
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
ok = True


def row(name, good, detail=""):
    global ok
    print(f"[{'OK ' if good else 'FAIL'}] {name} {detail}")
    ok &= bool(good)


def warn(name, detail=""):
    print(f"[WARN] {name} {detail}")


row("python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0])
for m in ("fastapi", "pydantic", "sqlalchemy", "alembic", "aiokafka", "prometheus_client", "cv2", "numpy", "yaml", "cryptography"):
    try:
        row(f"import {m}", True, getattr(importlib.import_module(m), "__version__", ""))
    except Exception as e:  # noqa: BLE001
        row(f"import {m}", False, str(e))
for m in ("torch", "ultralytics", "pynvml"):
    try:
        mod = importlib.import_module(m)
        print(f"[OK ] optional {m} {getattr(mod, '__version__', '')}")
    except Exception:
        warn(f"optional {m} not installed", "(needed only for GPU/YOLO workers)")
from app.ai.model_manager import cuda_status  # noqa: E402
from app.utils import ffmpeg as ff  # noqa: E402

st = cuda_status()
print(f"[INFO] torch/cuda: {st}")
caps = ff.detect_capabilities()
row("ffmpeg found", caps.ffmpeg_found, caps.version)
row("libx264 encoder", "libx264" in caps.encoders)
print(f"[INFO] hwaccels={caps.hwaccels}\n[INFO] NVDEC possible (hwaccel cuda listed AND NVIDIA device node present): {caps.nvdec_possible}")
print(f"[INFO] NVENC trial-encode results: {caps.nvenc_usable or 'none (no *_nvenc encoder compiled in)'}")
row("ffprobe found", bool(shutil.which("ffprobe")))
if os.environ.get("VMS_REQUIRE_CUDA", "").lower() in ("1", "true"):
    row("CUDA available (VMS_REQUIRE_CUDA=true)", st["cuda"])
sys.exit(0 if ok else 1)
