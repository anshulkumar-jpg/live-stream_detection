"""FFmpeg/FFprobe command construction and capability detection.

NVDEC/NVENC availability is *detected* (build flags, and for NVENC an actual 1-frame trial encode),
never assumed from the presence of an NVIDIA GPU.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import parse_qs, urlsplit

from app.core.config import DecodeCfg, EncoderCfg
from app.core.security import redact_text

log = logging.getLogger(__name__)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class FFmpegUnavailable(RuntimeError):
    pass


@dataclass
class Capabilities:
    ffmpeg_found: bool = False
    version: str = ""
    hwaccels: list[str] = field(default_factory=list)
    encoders: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    nvenc_usable: dict[str, bool] = field(default_factory=dict)  # "h264_nvenc": trial-encode result
    nvdec_possible: bool = False  # hwaccel cuda listed in build; real use is validated at runtime

    def can_encode(self, name: str) -> bool:
        return name in self.encoders and (not name.endswith("_nvenc") or self.nvenc_usable.get(name, False))


def _run(args: list[str], timeout: float = 15) -> str:
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout, text=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise FFmpegUnavailable(str(e)) from e
    return (r.stdout or "") + (r.stderr if r.returncode else "")


def _trial_encode(codec: str) -> bool:
    try:
        r = subprocess.run(
            [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:r=5",
             "-frames:v", "2", "-c:v", codec, "-f", "null", "-"],
            capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@lru_cache(maxsize=1)
def detect_capabilities() -> Capabilities:
    caps = Capabilities()
    if not shutil.which(FFMPEG):
        return caps
    caps.ffmpeg_found = True
    try:
        caps.version = _run([FFMPEG, "-hide_banner", "-version"]).splitlines()[0]
        caps.hwaccels = [l.strip() for l in _run([FFMPEG, "-hide_banner", "-hwaccels"]).splitlines()[1:] if l.strip()]
        enc = _run([FFMPEG, "-hide_banner", "-encoders"])
        caps.encoders = [p.split()[1] for p in enc.splitlines() if p.startswith(" V")]
        flt = _run([FFMPEG, "-hide_banner", "-filters"])
        caps.filters = [p.split()[1] for p in flt.splitlines() if len(p.split()) > 2 and p[:2].strip() != ""] if flt else []
    except FFmpegUnavailable:
        return caps
    # build lists cuda hwaccel AND an NVIDIA device node is visible (Linux driver or WSL2). Final proof is
    # at runtime: CameraIngest falls back to CPU after repeated cuda-decoder failures when policy allows.
    import os

    dev = any(os.path.exists(p) for p in ("/dev/nvidiactl", "/dev/dxg"))
    caps.nvdec_possible = "cuda" in caps.hwaccels and dev
    for c in ("h264_nvenc", "hevc_nvenc"):
        if c in caps.encoders:
            caps.nvenc_usable[c] = _trial_encode(c)
    return caps


# ----------------------------------------------------------------------- probing
@dataclass
class ProbeResult:
    ok: bool
    width: int = 0
    height: int = 0
    codec: str = ""
    fps: float = 0.0
    error: str | None = None


def parse_synthetic(url: str) -> dict:
    q = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    return {"w": int(q.get("w", 1280)), "h": int(q.get("h", 720)), "fps": float(q.get("fps", 10)),
            "pattern": q.get("pattern", "testsrc2")}


def probe_stream(url: str, transport: str = "tcp", timeout_s: float = 10.0) -> ProbeResult:
    if url.startswith("synthetic://"):
        s = parse_synthetic(url)
        return ProbeResult(True, s["w"], s["h"], "synthetic", s["fps"])
    args = [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-select_streams", "v:0",
            "-protocol_whitelist", "rtsp,rtp,rtcp,udp,tcp,tls,crypto",
            "-rtsp_transport", transport, "-timeout", str(int(timeout_s * 1e6)), url]
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout_s + 5, text=True)
    except (OSError, subprocess.TimeoutExpired) as e:
        return ProbeResult(False, error=redact_text(str(e)))
    if r.returncode != 0:
        return ProbeResult(False, error=redact_text(r.stderr.strip()[-300:]))
    try:
        st = json.loads(r.stdout)["streams"][0]
        num, _, den = st.get("avg_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        return ProbeResult(True, int(st["width"]), int(st["height"]), st.get("codec_name", ""), fps)
    except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
        return ProbeResult(False, error=f"unparseable ffprobe output: {e}")


# ---------------------------------------------------------------------- decoding
def scale_filter(w: int, h: int, aspect: str) -> str:
    if aspect == "stretch":
        return f"scale={w}:{h}"
    if aspect == "crop":
        return f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    return f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"


@dataclass
class DecodePlan:
    args: list[str]
    decoder: str  # "cuda" | "cpu"


def choose_decoder(cfg: DecodeCfg, caps: Capabilities | None = None) -> str:
    caps = caps or detect_capabilities()
    if cfg.hwaccel == "none" or caps is None:
        return "cpu"
    if cfg.hwaccel in ("cuda", "auto") and caps.nvdec_possible:
        return "cuda"
    if cfg.hwaccel == "cuda" and not cfg.allow_cpu_fallback:
        raise FFmpegUnavailable("hwaccel=cuda requested, FFmpeg lacks cuda hwaccel and CPU fallback is disabled")
    if cfg.hwaccel == "cuda":
        log.warning("cuda hwaccel unavailable in FFmpeg build; falling back to CPU decode (policy allows)")
    return "cpu"


def build_decode_args(url: str, cfg: DecodeCfg, decoder: str, realtime_synthetic: bool = True) -> list[str]:
    """Return an argv producing raw bgr24 frames of cfg.width x cfg.height on stdout."""
    base = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin"]
    vf = scale_filter(cfg.width, cfg.height, cfg.aspect)
    out = ["-an", "-sn", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    if url.startswith("synthetic://"):
        s = parse_synthetic(url)
        src = f"{s['pattern']}=size={s['w']}x{s['h']}:rate={s['fps']}"
        pre = ["-re"] if realtime_synthetic else []
        return base + pre + ["-f", "lavfi", "-i", src] + out
    args = base + ["-protocol_whitelist", "rtsp,rtp,rtcp,udp,tcp,tls,crypto",
                   "-rtsp_transport", cfg.rtsp_transport,
                   "-timeout", str(int(cfg.read_timeout_s * 1e6)),
                   "-fflags", "+discardcorrupt+nobuffer", "-flags", "low_delay"]
    if decoder == "cuda":
        args += ["-hwaccel", "cuda"]  # frames are downloaded automatically for the CPU scale filter
    return args + ["-i", url] + out


# ---------------------------------------------------------------------- encoding
@dataclass
class EncodePlan:
    encoder_name: str
    kind: str  # nvenc | software
    codec_args: list[str]


def choose_encoder(cfg: EncoderCfg, caps: Capabilities | None = None) -> EncodePlan:
    caps = caps or detect_capabilities()
    nv, sw = ("h264_nvenc", "libx264") if cfg.codec == "h264" else ("hevc_nvenc", "libx265")
    use_nv = cfg.encoder in ("auto", "nvenc") and caps.can_encode(nv)
    if cfg.encoder == "nvenc" and not use_nv and not cfg.allow_software_fallback:
        raise FFmpegUnavailable(f"{nv} requested but not usable and software fallback is disabled")
    if use_nv:
        return EncodePlan(nv, "nvenc", ["-c:v", nv, "-preset", cfg.nvenc_preset, "-tune", "ll", "-rc", "cbr",
                                        "-b:v", f"{cfg.bitrate_kbps}k", "-g", str(cfg.gop), "-bf", "0"])
    if sw not in caps.encoders:
        raise FFmpegUnavailable(f"software encoder {sw} not present in this FFmpeg build")
    extra = ["-tune", "zerolatency"] if sw == "libx264" else ["-x265-params", "log-level=error"]
    return EncodePlan(sw, "software", ["-c:v", sw, "-preset", cfg.preset, *extra,
                                       "-b:v", f"{cfg.bitrate_kbps}k", "-maxrate", f"{cfg.bitrate_kbps}k",
                                       "-bufsize", f"{cfg.bitrate_kbps * 2}k", "-g", str(cfg.gop), "-bf", "0"])


def build_encode_args(plan: EncodePlan, cfg: EncoderCfg, sink: str, rtsp_transport: str = "tcp") -> list[str]:
    """Raw bgr24 on stdin -> encoded stream to `sink` (rtsp:// URL, or file path for tests)."""
    head = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{cfg.width}x{cfg.height}", "-r", str(cfg.fps), "-i", "-"]
    tail = ["-pix_fmt", "yuv420p"]
    if sink.startswith("rtsp://"):
        tail += ["-f", "rtsp", "-rtsp_transport", rtsp_transport, sink]
    elif sink == "null":
        tail += ["-f", "null", "-"]
    else:
        tail += ["-f", "mpegts" if sink.endswith(".ts") else "mp4", sink]
    return head + plan.codec_args + tail
