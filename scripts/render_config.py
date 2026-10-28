#!/usr/bin/env python3
"""Render ${VAR} templates with environment variables (used by the `config-render` compose service).
Usage: render_config.py SRC DST   (unset vars fall back to the defaults below; unknown ${X} are left as-is)"""
import os
import sys
from string import Template

DEFAULTS = {"WEBRTC_ALLOW_ORIGIN": "http://localhost:5173", "WEBRTC_ADDITIONAL_HOSTS": "",
            "MTX_API_USER": "api", "MTX_API_PASS": ""}
src, dst = sys.argv[1], sys.argv[2]
env = {**DEFAULTS, **{k: v for k, v in os.environ.items() if v != ""}}
open(dst, "w").write(Template(open(src).read()).safe_substitute(env))
print(f"rendered {src} -> {dst}")
