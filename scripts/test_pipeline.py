#!/usr/bin/env python3
"""Self-contained pipeline smoke test (no Kafka/DB/GPU needed): synthetic source -> mock model -> annotated H.264 file.
Exit 0 only if events were produced and the output video is decodable. Run: python scripts/test_pipeline.py [seconds]"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("VMS_FORCE_MOCK_MODELS", "true")
from app.core.config import Settings, load_models_config, load_pipeline_config  # noqa: E402
from app.messaging.kafka_producer import MemoryBus  # noqa: E402
from app.messaging.topics import Topics  # noqa: E402
from app.schemas.worker import CameraSpec  # noqa: E402
from app.workers.gpu_worker import GpuWorker  # noqa: E402

root = os.path.join(os.path.dirname(__file__), "..")
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 10
out = tempfile.mkdtemp()
s = Settings(bus="memory", standalone=True, worker_id="smoke-worker", force_mock_models=True)
bus = MemoryBus()
w = GpuWorker(s, load_pipeline_config(f"{root}/config/pipeline.yaml"), load_models_config(f"{root}/config/models.yaml"), bus,
              lambda cid: f"{out}/{cid}.ts")
w.start(metrics_server=False)
w.start_camera(CameraSpec(camera_id="synthetic_001", source_url="synthetic://?w=1280&h=720&fps=15", model_profile="mock_full"), "smoke")
time.sleep(secs)
events = [v for t, _, v in list(bus.published) if t == Topics.DETECTION_EVENTS]
w.shutdown()
probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,nb_read_frames", "-of", "csv=p=0", f"{out}/synthetic_001.ts"],
                       capture_output=True, text=True).stdout.strip()
print(f"events={len(events)} output={out}/synthetic_001.ts ffprobe={probe}")
sys.exit(0 if events and probe.startswith("h264") else 1)
