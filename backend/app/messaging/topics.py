"""Topic names plus partition/retention configuration (documented in README)."""
from __future__ import annotations


class Topics:
    CAMERA_HEARTBEAT = "camera.heartbeat"
    WORKER_HEARTBEAT = "worker.heartbeat"
    DETECTION_EVENTS = "detection.events"
    PIPELINE_COMMANDS = "pipeline.commands"
    PIPELINE_METRICS = "pipeline.metrics"
    CAMERA_STATUS = "camera.status"

    ALL = (CAMERA_HEARTBEAT, WORKER_HEARTBEAT, DETECTION_EVENTS, PIPELINE_COMMANDS, PIPELINE_METRICS, CAMERA_STATUS)


# partitions: None => settings.kafka_topic_partitions. Keys: camera_id for per-camera ordering
# (detection.events, camera.heartbeat, camera.status); worker_id for worker topics/commands.
TOPIC_CONFIG: dict[str, dict] = {
    Topics.DETECTION_EVENTS: {"partitions": None, "retention_ms": 3 * 24 * 3600 * 1000, "key": "camera_id"},
    Topics.CAMERA_HEARTBEAT: {"partitions": None, "retention_ms": 3600 * 1000, "key": "camera_id", "cleanup": "delete"},
    Topics.WORKER_HEARTBEAT: {"partitions": 1, "retention_ms": 3600 * 1000, "key": "worker_id"},
    Topics.PIPELINE_COMMANDS: {"partitions": 1, "retention_ms": 24 * 3600 * 1000, "key": "worker_id"},
    Topics.PIPELINE_METRICS: {"partitions": 1, "retention_ms": 6 * 3600 * 1000, "key": "worker_id"},
    Topics.CAMERA_STATUS: {"partitions": None, "retention_ms": 24 * 3600 * 1000, "key": "camera_id"},
}
