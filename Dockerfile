# =============================================================================
# ACE-Step Studio — Universal Linux Dockerfile (CPU + GPU)
# =============================================================================
#
# Build:
#   docker build -t acestep-studio .
#
# Run:
#   docker run -d --name acestep -p 3001:3001 -p 7860:7860 -p 8001:8001 \
#     -v $(pwd)/checkpoints:/app/ACE-Step-1.5/checkpoints \
#     -v $(pwd)/data:/app/app/server/data \
#     --shm-size=2g acestep-studio
#
# With NVIDIA GPU (optional):
#   docker run -d --gpus all --name acestep -p 3001:3001 -p 7860:7860 -p 8001:8001 \
#     -v $(pwd)/checkpoints:/app/ACE-Step-1.5/checkpoints \
#     -v $(pwd)/data:/app/app/server/data \
#     --shm-size=2g acestep-studio
#
# =============================================================================

# ==================== Stage 1: Build frontend ====================
# Use full bookworm (not slim) — better-sqlite3 and esbuild need
# gcc, make, python3 for native module compilation.
FROM node:20-bookworm AS frontend-build

WORKDIR /build/app

# Copy package files first for layer caching
COPY app/package.json ./
COPY app/package-lock.json ./
RUN npm ci

# Build frontend
COPY app/ ./
RUN npx vite build

# Install server dependencies (better-sqlite3 compiles native .node addon)
WORKDIR /build/server
COPY app/server/package.json ./
COPY app/server/package-lock.json ./
RUN npm ci

# Copy server source
COPY app/server/ ./

# ==================== Stage 2: Runtime ====================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        build-essential \
        cmake \
        git \
        libsndfile1 \
        ffmpeg \
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir --break-system-packages --upgrade pip setuptools wheel

# ---- PyTorch CPU ----
RUN pip install --no-cache-dir --break-system-packages \
        torch==2.10.0 \
        torchvision==0.25.0 \
        torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cpu

# ---- Project source ----
WORKDIR /app

COPY ACE-Step-1.5/ /app/ACE-Step-1.5/

# Install nano-vllm from bundled source
RUN pip install --no-cache-dir --break-system-packages --no-deps \
        /app/ACE-Step-1.5/acestep/third_parts/nano-vllm

# ---- Python dependencies (required) ----
RUN pip install --no-cache-dir --break-system-packages \
        "safetensors==0.7.0" \
        "transformers>=4.51.0,<4.58.0" \
        "diffusers" \
        "gradio==6.2.0" \
        "matplotlib>=3.7.5" \
        "scipy>=1.10.1" \
        "soundfile>=0.13.1" \
        "loguru>=0.7.3" \
        "einops>=0.8.1" \
        "accelerate>=1.12.0" \
        "fastapi>=0.110.0" \
        "diskcache" \
        "uvicorn[standard]>=0.27.0" \
        "numba>=0.63.1" \
        "vector-quantize-pytorch>=1.27.15" \
        "toml" \
        "modelscope" \
        "peft>=0.18.0" \
        "lycoris-lora" \
        "lightning>=2.0.0" \
        "tensorboard>=2.20.0" \
        "typer-slim>=0.21.1" \
        "xxhash" \
        "pyyaml"

# ---- Optional packages (may fail on CPU — that is OK) ----
RUN pip install --no-cache-dir --break-system-packages "torchcodec>=0.9.1" || true
RUN pip install --no-cache-dir --break-system-packages "torchao" || true

# ---- Copy built frontend ----
COPY --from=frontend-build /build/app/dist /app/app/dist
COPY --from=frontend-build /build/server /app/app/server

# Copy remaining app files needed at runtime
COPY app/index.html app/favicon.svg app/metadata.json /app/app/
COPY app/services/ /app/app/services/
COPY app/data/ /app/app/data/
COPY app/context/ /app/app/context/
COPY app/i18n/ /app/app/i18n/

# ---- Runtime directories ----
RUN mkdir -p \
        /app/ACE-Step-1.5/checkpoints \
        /app/ACE-Step-1.5/gradio_outputs \
        /app/app/server/data

# ---- Environment ----
ENV ACESTEP_LLM_BACKEND=pt
ENV ACESTEP_API_HOST=0.0.0.0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_MODE=gradio
ENV ACESTEP_INIT_SERVICE=true
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ENV TOKENIZERS_PARALLELISM=false
ENV MANAGE_PIPELINE=true
ENV PORT=3001
ENV ACESTEP_PORT=8001
ENV PYTHON_PATH=python3
ENV ACESTEP_PATH=/app/ACE-Step-1.5
ENV NODE_ENV=production

EXPOSE 3001 7860 8001

HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:3001/ > /dev/null 2>&1 \
     || curl -sf http://localhost:7860/ > /dev/null 2>&1 \
     || exit 1

# ---- Entrypoint script ----
RUN printf '#!/bin/bash\n\
set -e\n\
echo "==========================================="\n\
echo "  ACE-Step Studio"\n\
echo "==========================================="\n\
echo "Python  : $(python --version 2>&1)"\n\
echo "Node.js : $(node --version 2>&1)"\n\
echo "PyTorch : $(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo N/A)"\n\
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then\n\
    echo "CUDA    : $(python -c "import torch; print(torch.version.cuda)")"\n\
    echo "GPU     : $(python -c "import torch; print(torch.cuda.get_device_name(0))")"\n\
else\n\
    echo "CUDA    : not available (CPU mode)"\n\
fi\n\
echo "==========================================="\n\
echo "Starting Express on 0.0.0.0:${PORT:-3001} ..."\n\
cd /app/app/server\n\
exec npx tsx src/index.ts\n' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
