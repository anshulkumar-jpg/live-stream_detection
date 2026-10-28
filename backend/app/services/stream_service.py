"""Playback URL generation, MediaMTX readiness and viewer-demand policy."""
from __future__ import annotations

import asyncio
import logging
import time

from app.core.config import Settings
from app.db.models import Camera
from app.schemas.camera import StreamInfo
from app.services.command_service import CommandService
from app.workers.mediamtx_publisher import MediaMTXClient, playback_urls

log = logging.getLogger(__name__)


class StreamService:
    def __init__(self, s: Settings, commands: CommandService):
        self.s, self.commands = s, commands
        self.mtx = MediaMTXClient(s.mediamtx_api_url, s.mediamtx_api_user, s.mediamtx_api_pass)
        self._leases: dict[str, float] = {}

    async def info(self, cam: Camera, touch: bool = True) -> StreamInfo:
        urls = playback_urls(self.s, cam.camera_id)
        note = None
        ready = False
        try:
            ready = await asyncio.to_thread(self.mtx.path_ready, cam.camera_id)
        except ConnectionError as e:
            note = str(e)
        hb = cam.heartbeat_detail or {}
        if touch and self.s.stream_policy == "on_demand" and cam.worker_id and cam.stream_enabled:
            if cam.camera_id not in self._leases:
                await self.commands.set_stream(cam.worker_id, cam.camera_id, True)
                note = "stream starting (on-demand); retry shortly"
            self._leases[cam.camera_id] = time.monotonic() + self.s.viewer_lease_s
        if not cam.stream_enabled:
            note = "streaming disabled for this camera"
        return StreamInfo(camera_id=cam.camera_id, path=cam.camera_id, ready=ready, publishing=bool(hb.get("stream_publishing")),
                          webrtc_url=urls["webrtc"], note=note)

    async def expire_leases(self, camera_lookup) -> None:
        """Stop encoding for on-demand cameras whose viewer lease expired."""
        t = time.monotonic()
        for cid, exp in list(self._leases.items()):
            if exp < t:
                del self._leases[cid]
                cam = await camera_lookup(cid)
                if cam and cam.worker_id:
                    await self.commands.set_stream(cam.worker_id, cid, False)
