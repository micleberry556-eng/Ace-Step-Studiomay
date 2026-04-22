# =============================================================================
# ACE-Step Studio — Universal Linux Dockerfile
# =============================================================================
#
# Builds the full ACE-Step Studio application for any Linux x86_64 system:
#   - Python ML pipeline (Gradio UI on port 7860 / REST API on port 8001)
#   - Node.js Express frontend (port 3001) serving the React web app
#
# Works with NVIDIA GPUs (CUDA 12.8) and falls back to CPU-only mode.
#
# Build:
#   docker build -t acestep-studio .
#
# Run (GPU — recommended):
#   docker run --gpus all -it --rm \
#     -p 3001:3001 -p 7860:7860 -p 8001:8001 \
#     -v $(pwd)/checkpoints:/app/ACE-Step-1.5/checkpoints \
#     -v $(pwd)/data:/app/app/server/data \
#     acestep-studio
#
# Run (CPU-only — no --gpus flag):
#   docker run -it --rm \
#     -p 3001:3001 -p 7860:7860 -p 8001:8001 \
#     -v $(pwd)/checkpoints:/app/ACE-Step-1.5/checkpoints \
#     -v $(pwd)/data:/app/app/server/data \
#     acestep-studio
#
# =============================================================================

# ==================== Stage 1: Node.js frontend build ====================
FROM node:20-bookworm-slim AS frontend-build

WORKDIR /build/app

# Install frontend dependencies
COPY app/package.json app/package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install

# Copy frontend source and build
COPY app/ ./
RUN npx vite build

# Install server dependencies
WORKDIR /build/app/server
COPY app/server/package.json app/server/package-lock.json* ./
RUN npm ci --ignore-scripts 2>/dev/null || npm install

# Copy server source
COPY app/server/ ./

# ==================== Stage 2: Runtime image ====================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Python
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        # Build tools (needed for some pip packages)
        build-essential \
        cmake \
        git \
        # Audio processing
        libsndfile1 \
        ffmpeg \
        # Node.js 20.x
        curl \
        ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Ensure python -> python3 symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- PyTorch with CUDA 12.8 ----
RUN pip install --no-cache-dir \
        torch==2.10.0+cu128 \
        torchvision==0.25.0+cu128 \
        torchaudio==2.10.0+cu128 \
        --extra-index-url https://download.pytorch.org/whl/cu128

# ---- Project source ----
WORKDIR /app

# Copy the ML pipeline
COPY ACE-Step-1.5/ /app/ACE-Step-1.5/

# Install nano-vllm from bundled source (before requirements to cache layers)
RUN pip install --no-cache-dir --no-deps \
        /app/ACE-Step-1.5/acestep/third_parts/nano-vllm

# Install Python dependencies (excluding torch/torchvision/torchaudio already installed)
RUN pip install --no-cache-dir \
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
        "torchcodec>=0.9.1" \
        "torchao" \
        "toml" \
        "modelscope" \
        "peft>=0.18.0" \
        "lycoris-lora" \
        "lightning>=2.0.0" \
        "tensorboard>=2.20.0" \
        "typer-slim>=0.21.1" \
        "xxhash" \
        "pyyaml" \
        "bitsandbytes>=0.49.0" \
        "triton>=3.0.0" \
        "flash-attn" \
    || echo "NOTE: Some optional packages (flash-attn) may fail on certain systems — this is OK"

# ---- Copy built frontend ----
COPY --from=frontend-build /build/app/dist /app/app/dist
COPY --from=frontend-build /build/app/server /app/app/server

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

# ---- Environment defaults ----
# Python ML pipeline
ENV ACESTEP_LLM_BACKEND=pt
ENV ACESTEP_API_HOST=0.0.0.0
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV ACESTEP_MODE=gradio
ENV ACESTEP_INIT_SERVICE=true
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Node.js Express server
ENV MANAGE_PIPELINE=true
ENV PORT=3001
ENV ACESTEP_PORT=8001
ENV PYTHON_PATH=python3
ENV ACESTEP_PATH=/app/ACE-Step-1.5
ENV NODE_ENV=production

# ---- Ports ----
# 3001 = Express web app | 7860 = Gradio UI | 8001 = REST API
EXPOSE 3001 7860 8001

# ---- Health check ----
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:3001/ > /dev/null 2>&1 \
     || curl -sf http://localhost:7860/ > /dev/null 2>&1 \
     || exit 1

# ---- Entrypoint ----
COPY <<'EOF' /app/docker-entrypoint.sh
#!/usr/bin/env bash
set -e

echo "==========================================="
echo "  ACE-Step Studio — Linux Container"
echo "==========================================="
echo "Python    : $(python --version 2>&1)"
echo "Node.js   : $(node --version 2>&1)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "CUDA      : $(python -c 'import torch; print(torch.version.cuda)')"
    echo "GPU       : $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "VRAM      : $(python -c 'import torch; p=torch.cuda.get_device_properties(0); print(f"{p.total_mem/1024**3:.1f} GB")' 2>/dev/null || echo 'unknown')"
else
    echo "CUDA      : NOT AVAILABLE — running on CPU"
    echo "           (use --gpus all for GPU acceleration)"
fi
echo "==========================================="

# Start Express server (which manages the Python pipeline as a child process)
echo "Starting Express server on 0.0.0.0:${PORT:-3001} ..."
cd /app/app/server
exec npx tsx src/index.ts
EOF

RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
