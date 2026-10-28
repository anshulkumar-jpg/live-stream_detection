"""MediaMTX helpers: publish URL construction, API client (stdlib only), URL generation for viewers."""
from __future__ import annotations

import base64
import json
import logging
import urllib.error
import urllib.request
from urllib.parse import quote

from app.core.config import Settings

log = logging.getLogger(__name__)


def build_publish_url(s: Settings, path: str) -> str:
    """RTSP publish URL (credentials embedded; only ever passed to the FFmpeg argv, always redacted in logs)."""
    cred = ""
    if s.mediamtx_publish_user:
        cred = f"{quote(s.mediamtx_publish_user, safe='')}:{quote(s.mediamtx_publish_pass, safe='')}@"
    return f"rtsp://{cred}{s.mediamtx_host}:{s.mediamtx_rtsp_port}/{path}"


def playback_urls(s: Settings, path: str) -> dict[str, str]:
    """Browser-facing URLs derived from configured public bases, which must mirror mediamtx.yml and the
    compose port mapping (MediaMTX default: WebRTC 8889)."""
    return {"webrtc": f"{s.public_webrtc_base.rstrip('/')}/{path}"}


class MediaMTXClient:
    def __init__(self, api_url: str, user: str = "", password: str = "", timeout: float = 3.0):
        self.api, self.timeout = api_url.rstrip("/"), timeout
        self._auth = None
        if user:
            self._auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()

    def _get(self, path: str) -> dict | None:
        req = urllib.request.Request(f"{self.api}{path}")
        if self._auth:
            req.add_header("Authorization", self._auth)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (fixed config URL)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ConnectionError(f"MediaMTX API unreachable: {e}") from e

    def path_info(self, name: str) -> dict | None:
        return self._get(f"/v3/paths/get/{quote(name, safe='')}")

    def path_ready(self, name: str) -> bool:
        info = self.path_info(name)
        return bool(info and info.get("ready"))

    def reader_count(self, name: str) -> int:
        info = self.path_info(name)
        return len(info.get("readers", [])) if info else 0

    def healthy(self) -> bool:
        try:
            self._get("/v3/config/global/get")
            return True
        except ConnectionError:
            return False
