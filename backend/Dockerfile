# One image for control plane and workers.
#   CPU / dev  : docker build -t vms-backend backend
#   GPU worker : docker build -t vms-backend-gpu --build-arg BASE_IMAGE=nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 \
#                  --build-arg WITH_GPU=1 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126 backend
# The host NVIDIA driver must support the CUDA runtime of the chosen base image / torch wheel.
ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}
ARG WITH_GPU=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=15 PIP_BREAK_SYSTEM_PACKAGES=1 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl \
      $(command -v python3 >/dev/null 2>&1 || echo python3 python3-pip) \
    && rm -rf /var/lib/apt/lists/* \
    && (command -v pip >/dev/null 2>&1 || ln -s "$(command -v pip3)" /usr/local/bin/pip) \
    && (command -v python >/dev/null 2>&1 || ln -s "$(command -v python3)" /usr/local/bin/python)

WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
RUN pip install -r requirements.txt \
    && if [ "$WITH_GPU" = "1" ]; then pip install torch torchvision --index-url "$TORCH_INDEX" && pip install -r requirements-gpu.txt; fi

# YOLOv5 weights (legacy checkpoints) cannot be loaded by ultralytics>=8: the `yolov5` package is installed
# without its pinned deps (they conflict with current torch/numpy) plus the few it imports at load time.
RUN if [ "$WITH_GPU" = "1" ]; then pip install --no-deps yolov5==7.0.13     && pip install "huggingface_hub==0.19.4" thop pandas seaborn matplotlib scipy tqdm PyYAML requests Pillow psutil; fi

COPY app ./app
COPY alembic.ini ./
RUN useradd -r -u 10001 vms && chown -R vms /app
USER vms
ENV PYTHONPATH=/app
# default: control plane. Workers override the command (see docker-compose.yml).
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app_factory --factory --host 0.0.0.0 --port 8000"]
