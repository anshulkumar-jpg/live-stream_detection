"""Settings (environment) and pipeline/model configuration (YAML)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VMS_", env_file=".env", extra="ignore")

    env: str = "dev"
    log_level: str = "INFO"

    # --- control plane
    database_url: str = "sqlite+aiosqlite:///./vms.db"
    db_pool_size: int = 10
    fernet_key: str = ""
    jwt_secret: str = ""
    auth_enabled: bool = True
    admin_username: str = "admin"
    admin_password: str = ""  # required when auth_enabled
    viewer_username: str = "viewer"
    viewer_password: str = ""
    rate_limit_per_sec: float = 20.0
    rate_limit_burst: int = 60
    cors_origins: list[str] = ["http://localhost:5173"]
    camera_allowed_cidrs: list[str] = []
    allow_synthetic_sources: bool = False

    # --- messaging
    bus: Literal["kafka", "memory"] = "kafka"
    kafka_bootstrap: str = "localhost:9092"
    kafka_client_id: str = "vms"
    kafka_acks: str = "all"
    kafka_retries: int = 5
    kafka_delivery_timeout_ms: int = 30000
    kafka_buffer_size: int = 5000
    kafka_topic_partitions: int = 6
    kafka_replication: int = 1

    # --- streaming
    mediamtx_host: str = "localhost"  # host the *worker* publishes to
    mediamtx_rtsp_port: int = 8554
    mediamtx_api_url: str = "http://localhost:9997"
    mediamtx_publish_user: str = "publisher"
    mediamtx_publish_pass: str = ""
    mediamtx_api_user: str = ""
    mediamtx_api_pass: str = ""
    # public (browser-facing) base URLs; MUST match mediamtx.yml + compose port mapping
    public_webrtc_base: str = "http://localhost:8889"
    public_hls_base: str = "http://localhost:8888"

    # --- control-plane policies
    worker_heartbeat_timeout_s: float = 15.0
    scheduler_interval_s: float = 5.0
    autoscaler_enabled: bool = False
    max_streams_per_worker: int = 16
    gpu_mem_high_pct: float = 90.0
    queue_backlog_high: int = 50
    p95_latency_high_s: float = 1.0
    command_ack_timeout_s: float = 20.0
    stream_policy: Literal["always", "on_demand"] = "always"
    viewer_lease_s: float = 120.0
    event_retention_days: int = 14

    # --- worker
    worker_id: str = "gpu-worker-0"
    gpu_id: int = 0
    require_cuda: bool = False
    force_mock_models: bool = False  # explicit dev switch: ALL models -> mock backend (logged, reported)
    synthetic_default: bool = False
    pipeline_config: str = "config/pipeline.yaml"
    models_config: str = "config/models.yaml"
    cameras_config: str = "config/cameras.yaml"
    worker_metrics_port: int = 9101
    standalone: bool = False  # run cameras.yaml directly without control plane

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.env == "prod":
            if not self.fernet_key or not self.jwt_secret:
                raise ValueError("prod requires VMS_FERNET_KEY and VMS_JWT_SECRET")
            if self.auth_enabled and not self.admin_password:
                raise ValueError("prod requires VMS_ADMIN_PASSWORD")
            if self.bus != "kafka":
                raise ValueError("prod requires Kafka bus")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ------------------------------------------------------------------ pipeline.yaml
class DecodeCfg(BaseModel):
    width: int = 1280
    height: int = 720
    aspect: Literal["stretch", "letterbox", "crop"] = "letterbox"
    hwaccel: Literal["auto", "cuda", "none"] = "auto"
    allow_cpu_fallback: bool = True
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    read_timeout_s: float = 10.0
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 30.0
    stable_after_s: float = 10.0  # backoff resets after this many seconds of good frames


class RouterCfg(BaseModel):
    queue_size: int = 4
    default_fps: float = 5.0
    stale_ms: int = 2000
    fair_burst: int = 2  # max consecutive frames from one camera in a batch


class InferenceCfg(BaseModel):
    batch_size: int = 8
    batch_timeout_ms: int = 25
    gpu_mem_limit_fraction: float = 0.9  # refuse to run new batches above this (soft guard)


class OverlayCfg(BaseModel):
    enabled: bool = True
    draw_labels: bool = True
    draw_timestamp: bool = True
    draw_confidence: bool = True
    thickness: int = 2
    detection_hold_ms: int = 1500
    class_colors: dict[str, tuple[int, int, int]] = {}  # BGR


class EncoderCfg(BaseModel):
    codec: Literal["h264", "h265"] = "h264"
    encoder: Literal["auto", "nvenc", "software"] = "auto"
    allow_software_fallback: bool = True
    width: int = 1280
    height: int = 720
    fps: int = 15
    bitrate_kbps: int = 2500
    gop: int = 30
    preset: str = "veryfast"  # software preset
    nvenc_preset: str = "p3"
    queue_size: int = 3
    max_restarts: int = 10
    renditions: list[int] = [480, 360, 240, 144]  # extra output heights (HLS quality ladder); [] = single stream only

    def rendition_bitrate(self, h: int) -> int:
        return self.bitrate_kbps if h >= self.height else max(100, int(self.bitrate_kbps * (h / self.height) ** 1.5))

    def rendition_width(self, h: int) -> int:
        return int(round(self.width * h / self.height / 2)) * 2

    def heights(self) -> list[int]:
        return [self.height] + sorted({h for h in self.renditions if 0 < h < self.height}, reverse=True)


class WorkerCfg(BaseModel):
    max_streams: int = 16
    heartbeat_interval_s: float = 3.0
    metrics_interval_s: float = 10.0
    drain_grace_s: float = 30.0


class PipelineConfig(BaseModel):
    worker: WorkerCfg = WorkerCfg()
    decode: DecodeCfg = DecodeCfg()
    router: RouterCfg = RouterCfg()
    inference: InferenceCfg = InferenceCfg()
    overlay: OverlayCfg = OverlayCfg()
    encoder: EncoderCfg = EncoderCfg()


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def load_pipeline_config(path: str | None = None) -> PipelineConfig:
    return PipelineConfig.model_validate(_load_yaml(path or get_settings().pipeline_config))


# ------------------------------------------------------------------- models.yaml
class ModelSpec(BaseModel):
    kind: Literal["detection", "ocr", "reid", "dino", "segmentation"]
    backend: str = "mock"  # mock | ultralytics | yolov5 | easyocr | torchreid | hf_dino | sam3 ...
    weights: str | None = None
    version: str = "unversioned"
    options: dict = Field(default_factory=dict)  # conf, iou, imgsz, half, classes, langs, ...


class ProfileSpec(BaseModel):
    detection: str
    stages: list[Literal["detect", "ocr", "reid", "dino", "sam"]] = ["detect"]
    ocr: str | None = None
    reid: str | None = None
    dino: str | None = None
    sam: str | None = None
    fps: float | None = None
    priority: int = 0  # higher scheduled first
    ocr_classes: list[str] = []
    reid_classes: list[str] = ["person"]
    sam_classes: list[str] = []
    sam_every_n: int = 5
    dino_every_n: int = 10
    overlay: OverlayCfg | None = None

    @model_validator(mode="after")
    def _stage_refs(self) -> "ProfileSpec":
        for st, ref in (("ocr", self.ocr), ("reid", self.reid), ("dino", self.dino), ("sam", self.sam)):
            if st in self.stages and not ref:
                raise ValueError(f"profile enables stage '{st}' but defines no '{st}' model reference")
        return self


class ModelsConfig(BaseModel):
    models: dict[str, ModelSpec] = {}
    profiles: dict[str, ProfileSpec] = {}

    @model_validator(mode="after")
    def _refs(self) -> "ModelsConfig":
        for pname, p in self.profiles.items():
            for ref in (p.detection, p.ocr, p.reid, p.dino, p.sam):
                if ref and ref not in self.models:
                    raise ValueError(f"profile '{pname}' references unknown model '{ref}'")
        return self

    def class_names(self, profile: str) -> list[str] | None:
        """Detection classes a user can pick for this profile (None = the model does not declare them)."""
        p = self.profiles.get(profile)
        names = self.models[p.detection].options.get("class_names") if p else None
        return [str(n) for n in names] if names else None


def load_models_config(path: str | None = None) -> ModelsConfig:
    return ModelsConfig.model_validate(_load_yaml(path or get_settings().models_config))


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.lower() in ("1", "true", "yes")
