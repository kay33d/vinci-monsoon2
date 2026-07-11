# =============================================================================
# SUBMISSION IMAGE — CPU-ONLY, linux/amd64.
#
# The grading VM is 4 GB RAM / 2 vCPU with NO GPU. Do NOT add CUDA or ROCm
# layers here: they would be dead weight against the 10 GB compressed cap and
# provide zero speedup. If you want GPU-accelerated *local development* on AMD
# hardware, create a separate Dockerfile.gpu-dev (ROCm base image + llama-cpp
# built with -DGGML_HIP=ON) — that file must NEVER be the submitted image.
#
# Build:  docker build --platform linux/amd64 -t hybrid-router:local .
# =============================================================================

# ---------- Stage 1: build llama-cpp-python wheel + fetch the GGUF ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Portable amd64 build: no -march=native (the grading CPU is unknown).
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# 2-3B 4-bit GGUF: ~2 GB file, fits the 4 GB RAM budget with headroom for
# Python + llama.cpp overhead. Override with:
#   docker build --build-arg MODEL_URL=<other-gguf-url> ...
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && curl -L --fail --retry 3 -o /models/model.gguf "$MODEL_URL"

# ---- Stage-1 classifier: DeBERTa router (torch CPU — never pull the CUDA
# wheel from PyPI; it alone would blow past the image budget) ---------------
RUN pip install --no-cache-dir --prefix=/install \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install \
        "transformers>=4.40" "huggingface_hub>=0.23" sentencepiece safetensors

# Bake the classifier snapshot into the image and VALIDATE it at build time:
# scripts/bake_router_model.py downloads the repo, makes it offline-loadable
# (backbone config saved locally), loads the weights through the exact
# runtime code path, and runs the 8-category smoke evaluation — the build
# fails below the 68% routing-accuracy target, so a bad snapshot never ships.
ARG ROUTER_MODEL_ID="xubayer/prompt-router-deberta-v3-xsmall"
COPY scripts/bake_router_model.py /tmp/bake/bake_router_model.py
COPY src/local_models/router_classifier.py /tmp/bake/router_classifier.py
RUN PYTHONPATH=/install/lib/python3.11/site-packages:/tmp/bake \
    python /tmp/bake/bake_router_model.py --model-id "$ROUTER_MODEL_ID" --out /models/router

# ---------- Stage 2: slim runtime -------------------------------------------
FROM python:3.11-slim

# libgomp is the only extra runtime lib llama.cpp needs on CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /models /models

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py .

ENV LOCAL_MODEL_PATH=/models/model.gguf \
    ROUTER_MODEL_PATH=/models/router \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are injected by the
# grading harness at runtime — never baked into the image.
ENTRYPOINT ["python", "entrypoint.py"]
