# VMS — Distributed real-time AI video surveillance pipeline

RTSP cameras → FFmpeg decode (NVDEC when usable) → per-camera bounded frame router → **shared** batched model pool
(YOLO / OCR / ReID / DINO / SAM adapters) → results → overlay + FFmpeg encode (NVENC when usable) → MediaMTX
(WebRTC) → any WebRTC player. A FastAPI control plane (PostgreSQL, Kafka) assigns cameras to GPU workers.

```
 cameras ──RTSP──► gpu-worker-N ── RTSP publish ──► MediaMTX ──WebRTC──► browser
                    │  ingest(1/camera) → router → inference scheduler (1 loop, shared models, batch 8)
                    │      → result processor (tracker, ReID gallery, events) → overlay+encoder (per camera)
                    ├── Kafka: detection.events, camera/worker.heartbeat, camera.status, pipeline.metrics
                    └── /metrics :9101 (Prometheus)
 FastAPI control plane ◄── Kafka ──► workers   (commands: pipeline.commands; no direct process control)
   └─ PostgreSQL (desired state, assignments, sessions, events)  ·  Prometheus  ·  Grafana
```

## Implementation status (read this first)

Verified here means "executed in the development sandbox: CPU only, no Docker daemon, no Kafka/Postgres/MediaMTX broker".

| Area | Status |
|---|---|
| FFmpeg ingest (RTSP/synthetic), reconnect/backoff, stall watchdog, credential redaction | **Implemented + tested** (synthetic source; bad-RTSP failure isolation tested; real cameras **not** tested) |
| NVDEC decode / NVENC encode | **Implemented, capability-detected, NOT verified** (no GPU here). NVENC is only chosen after a real trial encode; NVDEC needs the FFmpeg `cuda` hwaccel *and* an NVIDIA device node, with automatic CPU fallback if policy allows |
| Frame router (FPS sampling, stale/overflow drop, priority, fairness) | **Implemented + tested** |
| Shared inference scheduler (one loop, one model instance per worker, cross-camera batching) | **Implemented + tested** (with a counting test detector) |
| YOLO (Ultralytics) adapter | **Implemented, NOT verified** — needs `requirements-gpu.txt` and your weights; fails clearly if missing |
| Mock models (blob detector, OCR, ReID, DINO, SAM) | **Implemented + tested**; development only, always reported as `backend: mock` |
| OCR / ReID / DINO / **SAM3** real engines | **Interfaces + registry only.** Selecting `easyocr/paddleocr/torchreid/osnet/hf_dino/sam3…` raises `ModelUnavailable` with setup instructions (see below). No model is ever substituted silently |
| Overlay (boxes, labels, track ids, OCR text, timestamp, per-class colours) | **Implemented + tested** (pixel-level tests) |
| Encoder (H.264/H.265, restart on failure, constant output FPS) | **Implemented; H.264 software path tested**; H.265/NVENC untested |
| MediaMTX integration (publisher URLs, API client, HTTP-auth endpoint, config) | **Implemented; auth endpoint tested; MediaMTX itself NOT run here** (binary download blocked). WebRTC (WHEP) playback in the browser is **unverified** against a live MediaMTX |
| Kafka producer/consumer (bounded, non-blocking, retry, at-least-once) | **Implemented; non-blocking/bounded behaviour tested against a dead broker; real broker NOT tested** (pipeline tests use the in-memory bus with identical semantics) |
| Control plane (CRUD, start/stop, admission, scheduling, migration, fencing, reassignment, drain) | **Implemented + tested** against fake workers speaking the real command protocol |
| PostgreSQL | Alembic migration + models **tested on SQLite** (incl. model/migration drift check). Postgres-specific paths (JSONB variant, `ON CONFLICT`, asyncpg) are **not executed here** |
| Prometheus metrics, 6 Grafana dashboards | Metrics **tested**; dashboards **syntax-validated only** (not loaded in Prometheus/Grafana) |
| Docker Compose | **Written; `docker compose config` validates for both profiles; stack NOT started** (no Docker daemon here; image tags unverified against registries) |
| Autoscaling | Hook + recommendation only, **disabled by default**; no cloud integration |

No benchmark numbers, camera capacity, or GPU throughput claims are made. `max_streams: 16` is a starting target to validate on your hardware.

## Quick start

### 0. Prerequisites
Docker + Compose v2. For GPU workers: Linux, NVIDIA driver, NVIDIA Container Toolkit
(`docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi` must work).

### 1. Configure
```bash
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"   # → VMS_FERNET_KEY
# set all change-me-* passwords. Keep VMS_PUBLIC_*_BASE equal to the host ports published by compose.
docker compose --env-file .env config >/dev/null && echo compose-ok
```

### 2a. CPU-only development (synthetic camera + mock models, no GPU)
Set `VMS_REQUIRE_CUDA=false`, `VMS_FORCE_MOCK_MODELS=true`, `VMS_ALLOW_SYNTHETIC_SOURCES=true` in `.env`, then:
```bash
docker compose --profile dev up --build api worker-dev-0 kafka postgres mediamtx prometheus grafana
# create a synthetic camera (VMS_ALLOW_SYNTHETIC_SOURCES=true in .env):
curl -s localhost:8000/api/v1/cameras -H 'content-type: application/json' \
  -d '{"camera_id":"synthetic_001","name":"Synthetic","rtsp_url":"synthetic://?w=1280&h=720&fps=15","model_profile":"mock_full"}'
```
API docs: http://localhost:8000/docs · Grafana: http://localhost:3000 · Prometheus: http://127.0.0.1:9090

### 2b. Linux + NVIDIA GPU
Put weights in `./models/` (`yolo_general.pt`, or edit `config/models.yaml`), then:
```bash
docker compose up --build     # GPU is the default: gpu-worker-0 pinned to host GPU 0, VMS_REQUIRE_CUDA=true
docker compose exec gpu-worker-0 python /app/scripts/test_gpu.py     # CUDA visible inside the container?
```
Add `gpu-worker-1`: copy the service, set `VMS_WORKER_ID=gpu-worker-1`, `device_ids: ["1"]` (inside the container it is still `cuda:0`, so keep `VMS_GPU_ID=0`).
With `VMS_REQUIRE_CUDA=true` the worker **refuses to start** (exit 2, clear message) when CUDA isn't visible.

### 2c. Windows
Docker Desktop with the WSL2 backend is fine for `--profile dev`. GPU passthrough on Docker Desktop needs a current NVIDIA
Windows driver with WSL2 CUDA support; NVENC/NVDEC availability there is not guaranteed and is unverified here. Validate with
`docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu24.04 nvidia-smi` and `scripts/check_environment.py` inside the worker
before relying on the `gpu` profile. Production deployment is documented/targeted for Linux.

### 3. Without Docker (development)
```bash
cd backend && pip install -r requirements-dev.txt          # Python ≥ 3.12, FFmpeg + ffprobe on PATH
python ../scripts/check_environment.py                     # what this machine can really do (NVDEC/NVENC are *tested*)
python ../scripts/test_pipeline.py 10                      # synthetic → mock model → annotated H.264 file (exit 0 = OK)
VMS_FORCE_MOCK_MODELS=true VMS_BUS=memory python -m app.workers.gpu_worker --standalone --sink-dir /tmp/out   # Phase-2 single camera
python -m pytest                                            # full suite (~2 min; FFmpeg integration tests are marked `slow`)
```

## Repository layout
The React frontend was removed by request; the API is consumed directly (OpenAPI at `/docs`, WebSocket `/api/v1/ws`).

`backend/app/{api,core,schemas,db,services,messaging,monitoring,workers,ai,utils}` as specified, plus `ai/base.py` (adapter
interfaces/registry) and `services/container.py` (wiring + control loop). `config/` holds `pipeline.yaml`, `models.yaml`,
`cameras.yaml` (standalone mode only), `mediamtx.yml` (template) and `prometheus.yml` (template). `monitoring/` has
Grafana provisioning; regenerate dashboards with `python scripts/gen_dashboards.py`.

## Key behaviours

* **Single source of truth**: PostgreSQL holds *desired* state (`desired_state`, assignment, session); workers report *actual*
  state through `camera.status` / `camera.heartbeat`. The control plane never touches a worker's processes: it publishes
  idempotent commands (`start_camera`, `stop_camera`, `update_camera`, `set_stream`, `drain_worker`) with a `session_id`.
* **No duplicate sessions**: a camera has at most one `worker_id`. Migration = stop on source (await ack) → clear assignment →
  start on target with a new session. If a worker reports a camera that isn't assigned to it, the control plane sends it a
  `stop_camera` (fencing). A restarted worker (empty state) gets a fresh session; the old session is closed.
* **Admission control** (DB-counted, not heartbeat-counted): healthy + `ready`, below capacity, GPU memory, queue backlog, P95 latency.
  Cameras that can't be placed stay `pending` with the reason and are placed when capacity appears.
* **Latency control**: per-camera queue (default 4) drops the *oldest* frame; frames older than `stale_ms` are dropped at dequeue;
  `gpu_mem_limit_fraction` drops batches instead of risking OOM; inference never waits on Kafka (non-blocking publish).
* **Streaming**: encoder runs at constant FPS from the newest decoded frame with the latest detections held for `detection_hold_ms`.
  `VMS_STREAM_POLICY=always|on_demand`; on-demand starts encoding when `GET /cameras/{id}/stream` is called and stops after
  `VMS_VIEWER_LEASE_S`. Events keep flowing when streaming is disabled.
* **URLs** returned by the API come from `VMS_PUBLIC_WEBRTC_BASE`; it must match your published ports.
* **Metrics are process-local.** Cluster totals must be `sum by (...)` in PromQL (the dashboards do this). The `worker_heartbeat_age_seconds`
  series comes from the control plane's view.

## Kafka topics
| Topic | Key | Partitions | Retention | Content |
|---|---|---|---|---|
| `detection.events` | camera_id | `VMS_KAFKA_TOPIC_PARTITIONS` (6) | 3 d | `DetectionEvent` (no frames, no embeddings; `embedding_ref` pointers only) |
| `camera.heartbeat` | camera_id | 6 | 1 h | per-camera state and FPS/queue/drops |
| `worker.heartbeat` | worker_id | 1 | 1 h | GPU, latency quantiles, capabilities, readiness |
| `pipeline.commands` | worker_id | 1 | 24 h | control commands (**contains camera URLs with credentials — restrict ACLs to the API and workers**) |
| `pipeline.metrics` | worker_id | 1 | 6 h | FPS and latency summary |
| `camera.status` | camera_id | 6 | 24 h | actual-state changes + command acks |

Topics are created by the API at startup (`auto.create.topics.enable=false`). Delivery: `acks=all`, idempotent producer, retries,
bounded buffer (`VMS_KAFKA_BUFFER_SIZE`): when full, **new events are dropped and counted** (`kafka_publish_errors_total{reason="buffer_full"}`);
commands bypass the buffer and await delivery. Consumers use at-least-once; `idempotency_key` + a DB unique index de-duplicate events.
Optional snapshots belong in object storage; events would carry `snapshot_ref` (no snapshot writer is implemented).

## Security
RTSP URLs are Fernet-encrypted at rest and never returned by the API (only `rtsp://***@host/...`); logs, FFmpeg stderr and `repr()` are redacted
(tested). Camera IDs and URLs are validated (scheme allow-list, no whitespace/leading `-`, optional trusted CIDRs via
`VMS_CAMERA_ALLOWED_CIDRS`); FFmpeg is always spawned with an argv list and an RTSP protocol whitelist. JWT auth with `admin`/`viewer` roles (`POST /api/v1/auth/login`; admin can add, change, start, stop and delete cameras,
viewer is read-only), bcrypt-hashed accounts from `.env`, per-IP token-bucket rate limit. MediaMTX authentication is delegated to
`/api/v1/streams/mediamtx-auth` (publisher credentials for workers, a valid JWT to watch streams, separate API credentials).
Postgres and Kafka publish no host ports; Prometheus binds to localhost. **Before production:** TLS everywhere (reverse proxy for API/UI/WebRTC),
Kafka SASL/ACLs, per-service DB roles, restrict MediaMTX 8554/9997/9998 to the internal network, define recording/retention policy
(`VMS_EVENT_RETENTION_DAYS` purges event rows only; no video is recorded), pin and scan images, put the Fernet key in a secrets manager, add authentication.

## Model weights and adapters
* **YOLO**: place weights under `./models`, reference them in `config/models.yaml` (`weights:`), `conf/iou/imgsz/half` in `options`. Other
  detection models are just additional `models:` entries plus a `profiles:` entry. Nothing is downloaded.
* **Adding OCR / ReID / DINO / SAM3**: implement the interface in `app/ai/base.py` (`OcrEngine.read`, `ReidModel.embed`, `DinoModel.embed`,
  `Segmenter.segment`), then `register("<kind>", "<backend>", factory)` at import time (e.g. in `app/ai/<your_module>.py`, imported by `model_manager.py`),
  and reference it in `models.yaml`. **SAM3**: obtain weights and license from the publisher and confirm the package API; the adapter contract is
  box-prompted masks and runs only every `sam_every_n` frames on `sam_classes`. Embeddings stay in the in-process `EmbeddingStore`
  (bounded LRU); events carry references/dimensions only. ReID similarity means "looks like track N on this camera", never a global identity.
* `VMS_FORCE_MOCK_MODELS=true` maps every model to the mock backend explicitly (logged per model and shown in worker heartbeats).

## Operations
* `GET /ready` (DB + bus), `/health`, `/metrics`; workers expose `:9101/metrics` and publish heartbeats every 3 s.
* Graceful shutdown: SIGTERM → status `stopping`, cameras stopped (decoder/encoder subprocesses terminated), final events flushed.
* Troubleshooting: `scripts/check_environment.py`, `scripts/test_rtsp.py <url>`, `scripts/test_mediamtx.py [camera_id]`, `scripts/test_gpu.py`.

## Tests
`cd backend && python -m pytest` — unit (security, schemas, router, overlay, models), FFmpeg integration (ingest, reconnect, isolation, encoder restart,
shared batching, full worker pipeline, command protocol, shutdown), control plane (API, auth, admission, reassignment, migration, fencing, events,
streams, rate limit), infra (Alembic, Kafka non-blocking, metrics). Not covered (needs real infrastructure): live RTSP cameras, GPU/NVDEC/NVENC,
YOLO inference, real Kafka/Postgres/MediaMTX/Grafana, WebRTC playback, multi-GPU scheduling on real hardware.

## Known limitations
* Capture timestamps are wall-clock at frame read (raw pipe carries no PTS); add RTCP/NTP-based timestamps if you need frame-accurate time.
* Command topic carries credentials (see Security); a secret-reference scheme (Vault ID instead of URL) is the better production design.
* Single Kafka node and single Postgres in Compose are dev topology. The `heal_missing_sessions`/rebalance logic is conservative (cooldown 60 s) and untested under real failure timing.
