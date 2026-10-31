"""Security helpers: credential redaction, URL/ID validation, secret encryption, auth, rate limiting."""
from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_USERINFO_RE = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s]+@")  # greedy: a password may itself contain "@"
_TOKEN_QS_RE = re.compile(r"([?&]token=)[^&\s\"']+")
ALLOWED_PUBLIC_SCHEMES = {"rtsp", "rtsps"}
# Dev/test-only schemes. Enabled explicitly through Settings.allow_synthetic_sources.
DEV_SCHEMES = {"synthetic"}


class ValidationError(ValueError):
    pass


# --------------------------------------------------------------------- redaction
def redact_text(text: str) -> str:
    """Strip user:password@ from any URL found in free text (logs, ffmpeg stderr)."""
    return _TOKEN_QS_RE.sub(r"\1***", _USERINFO_RE.sub(r"\1***@", text))  # also hides ?token=<jwt> in access logs


def redact_url(url: str) -> str:
    return redact_text(url)


# ------------------------------------------------------------------- validation
def validate_camera_id(camera_id: str) -> str:
    if not CAMERA_ID_RE.match(camera_id or ""):
        raise ValidationError("camera_id must match ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    return camera_id


def validate_source_url(
    url: str,
    allowed_cidrs: Iterable[str] = (),
    allow_synthetic: bool = False,
    resolve: bool = True,
) -> str:
    """Validate a camera source URL. Raises ValidationError. Returns the URL unchanged.

    Blocks argument injection (whitespace, control chars, leading '-'), non-RTSP schemes and,
    when CIDRs are configured, hosts outside the trusted camera subnets.
    """
    if not url or len(url) > 2048:
        raise ValidationError("source URL missing or too long")
    if any(c.isspace() or ord(c) < 32 for c in url) or url.startswith("-"):
        raise ValidationError("source URL contains illegal characters")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    allowed = set(ALLOWED_PUBLIC_SCHEMES) | (DEV_SCHEMES if allow_synthetic else set())
    if scheme not in allowed:
        raise ValidationError(f"scheme '{scheme}' not allowed; allowed: {sorted(allowed)}")
    if scheme == "synthetic":
        return url
    if not parts.hostname:
        raise ValidationError("source URL has no host")
    cidrs = [ipaddress.ip_network(c, strict=False) for c in allowed_cidrs if c]
    if cidrs:
        try:
            addrs = {ipaddress.ip_address(parts.hostname)}
        except ValueError:
            if not resolve:
                raise ValidationError("hostname not allowed when CIDR restriction active and resolve disabled")
            try:
                addrs = {ipaddress.ip_address(i[4][0]) for i in socket.getaddrinfo(parts.hostname, None)}
            except OSError as e:
                raise ValidationError(f"cannot resolve host: {e}") from e
        if not all(any(a in net for net in cidrs) for a in addrs):
            raise ValidationError("camera host is outside trusted camera subnets")
    return url


# ------------------------------------------------------------------ encryption
class SecretBox:
    """Fernet encryption for stored RTSP URLs. Key must come from env/secret manager."""

    def __init__(self, key: str):
        if not key:
            raise ValueError("VMS_FERNET_KEY is required (generate with Fernet.generate_key())")
        self._f = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> str:
        return self._f.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        try:
            return self._f.decrypt(token.encode()).decode()
        except InvalidToken as e:
            raise ValueError("cannot decrypt secret: wrong key or corrupt data") from e


# ----------------------------------------------------------------------- auth
# jose/bcrypt are imported lazily so GPU workers (which never authenticate users) don't need them installed.
def hash_password(password: str) -> str:
    import bcrypt

    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(subject: str, role: str, secret: str, ttl_minutes: int = 60) -> str:
    from jose import jwt

    exp = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    return jwt.encode({"sub": subject, "role": role, "exp": exp}, secret, algorithm="HS256")


def decode_token(token: str, secret: str) -> dict:
    from jose import JWTError, jwt

    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError as e:
        raise ValidationError("invalid or expired token") from e


# ---------------------------------------------------------------- rate limiter
class TokenBucketLimiter:
    """Simple per-key in-memory limiter (per API process; use a gateway for cluster-wide limits)."""

    def __init__(self, rate_per_sec: float, burst: int):
        self.rate, self.burst = rate_per_sec, burst
        self._b: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._b.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._b[key] = (tokens, now)
                return False
            self._b[key] = (tokens - 1, now)
            if len(self._b) > 10000:  # bound memory
                self._b = dict(list(self._b.items())[-5000:])
            return True


class LoginThrottle:
    """Brute-force guard: after `max_fail` failed logins for one (client, username) within `window_s`, further
    attempts get 429 until the oldest failure ages out. In-memory, per API process."""

    def __init__(self, max_fail: int = 5, window_s: float = 300.0):
        self.max_fail, self.window_s = max_fail, window_s
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> int:
        now = time.monotonic()
        with self._lock:
            fails = [t for t in self._fails.get(key, []) if now - t < self.window_s]
            self._fails[key] = fails
            return int(self.window_s - (now - fails[0])) + 1 if len(fails) >= self.max_fail else 0

    def failed(self, key: str) -> None:
        with self._lock:
            self._fails.setdefault(key, []).append(time.monotonic())
            if len(self._fails) > 10000:  # bound memory
                self._fails = dict(list(self._fails.items())[-5000:])

    def succeeded(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)
