#!/usr/bin/env python3
"""Probe an RTSP source (credentials are redacted in output) and grab frames.
Usage: python scripts/test_rtsp.py rtsp://user:pass@host:554/stream1 [--frames 30] [--transport tcp|udp]"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.core.config import DecodeCfg  # noqa: E402
from app.core.security import redact_url, validate_source_url  # noqa: E402
from app.utils import ffmpeg as ff  # noqa: E402
from app.workers.camera_ingest import CameraIngest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--transport", default="tcp")
ap.add_argument("--allow-synthetic", action="store_true")
a = ap.parse_args()
validate_source_url(a.url, allow_synthetic=a.allow_synthetic)
print("source:", redact_url(a.url))
p = ff.probe_stream(a.url, a.transport)
print("probe:", "OK" if p.ok else "FAIL", f"{p.width}x{p.height} {p.codec} {p.fps:.1f}fps" if p.ok else p.error)
if not p.ok:
    sys.exit(1)
n, t0 = [0], time.time()
ing = CameraIngest("probe", a.url, "probe", DecodeCfg(rtsp_transport=a.transport), lambda f: n.__setitem__(0, n[0] + 1))
ing.start()
while n[0] < a.frames and time.time() - t0 < 30:
    time.sleep(0.2)
ing.stop()
dt = time.time() - t0
print(f"decoded {n[0]} frames in {dt:.1f}s ({n[0] / dt:.1f} fps) decoder={ing.decoder} state_error={ing.last_error}")
sys.exit(0 if n[0] >= a.frames else 2)
