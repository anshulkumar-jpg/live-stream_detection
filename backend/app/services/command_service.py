"""Defined command interface to workers (Kafka `pipeline.commands`). The control plane never touches
a worker's processes directly."""
from __future__ import annotations

import asyncio
import logging

from app.core.config import Settings
from app.messaging.kafka_producer import EventBus
from app.messaging.topics import Topics
from app.monitoring import metrics
from app.schemas.worker import CameraSpec, Command

log = logging.getLogger(__name__)


class CommandService:
    def __init__(self, bus: EventBus, settings: Settings):
        self.bus, self.s = bus, settings
        self._pending: dict[str, asyncio.Future] = {}

    async def send(self, cmd: Command, wait_ack: bool = False) -> str | None:
        """Publish a command. Returns an error string if the worker reported failure / timed out
        (only when wait_ack), or if publishing failed; else None."""
        fut = None
        if wait_ack:
            fut = asyncio.get_running_loop().create_future()
            self._pending[cmd.command_id] = fut
        try:
            await self.bus.publish_critical(Topics.PIPELINE_COMMANDS, cmd.worker_id, cmd.model_dump(mode="json"))
        except Exception as e:  # noqa: BLE001
            self._pending.pop(cmd.command_id, None)
            metrics.api_commands_total.labels(cmd.type, "publish_failed").inc()
            return f"command publish failed: {e}"
        metrics.api_commands_total.labels(cmd.type, "published").inc()
        if fut is None:
            return None
        try:
            return await asyncio.wait_for(fut, self.s.command_ack_timeout_s)
        except asyncio.TimeoutError:
            metrics.api_commands_total.labels(cmd.type, "ack_timeout").inc()
            return "worker did not acknowledge command in time"
        finally:
            self._pending.pop(cmd.command_id, None)

    def on_ack(self, command_id: str | None, state: str, error: str | None) -> None:
        fut = self._pending.get(command_id or "")
        if fut and not fut.done():
            fut.set_result(error if state == "error" else None)

    # convenience builders
    async def start_camera(self, worker_id: str, spec: CameraSpec, session_id: str, wait_ack: bool = False) -> str | None:
        return await self.send(Command(type="start_camera", worker_id=worker_id, camera_id=spec.camera_id,
                                       session_id=session_id, camera=spec), wait_ack)

    async def update_camera(self, worker_id: str, spec: CameraSpec, session_id: str) -> str | None:
        return await self.send(Command(type="update_camera", worker_id=worker_id, camera_id=spec.camera_id,
                                       session_id=session_id, camera=spec))

    async def stop_camera(self, worker_id: str, camera_id: str, wait_ack: bool = False) -> str | None:
        return await self.send(Command(type="stop_camera", worker_id=worker_id, camera_id=camera_id), wait_ack)

    async def set_stream(self, worker_id: str, camera_id: str, enabled: bool) -> str | None:
        return await self.send(Command(type="set_stream", worker_id=worker_id, camera_id=camera_id,
                                       params={"enabled": enabled}))

    async def drain(self, worker_id: str) -> str | None:
        return await self.send(Command(type="drain_worker", worker_id=worker_id))
