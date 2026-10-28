#!/usr/bin/env python3
"""Check MediaMTX API/ports and whether a camera path is publishing. Uses env: VMS_MEDIAMTX_API_URL, VMS_MEDIAMTX_API_USER/PASS.
Usage: python scripts/test_mediamtx.py [camera_id]"""
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.workers.mediamtx_publisher import MediaMTXClient  # noqa: E402

api = os.environ.get("VMS_MEDIAMTX_API_URL", "http://localhost:9997")
c = MediaMTXClient(api, os.environ.get("VMS_MEDIAMTX_API_USER", ""), os.environ.get("VMS_MEDIAMTX_API_PASS", ""))
print("MediaMTX API reachable:", c.healthy(), api)
host = os.environ.get("MTX_HOST", "localhost")
for name, port in (("rtsp", 8554), ("webrtc(http)", 8889)):
    s = socket.socket(); s.settimeout(1.5)
    try:
        s.connect((host, port)); print(f"port {port} ({name}) open")
    except OSError as e:
        print(f"port {port} ({name}) CLOSED: {e}")
    finally:
        s.close()
if len(sys.argv) > 1:
    try:
        print(f"path {sys.argv[1]} ready:", c.path_ready(sys.argv[1]))
    except ConnectionError as e:
        print(e); sys.exit(1)
