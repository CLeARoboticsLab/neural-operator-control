# ==============================================================================
# Multi-Task Operator Learning for Control — Reproducibility Container
# ==============================================================================
# Base: NVIDIA CUDA for JAX GPU support
# Includes: JAX, Equinox, CasADi/IPOPT
# ==============================================================================

FROM nvidia/cuda:12.3.1-cudnn9-devel-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    unzip \
    # CasADi / IPOPT build dependencies
    coinor-libipopt-dev \
    gfortran \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ---------- Application code + Python dependencies ----------
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[cuda]"

# ---------- Default entrypoint ----------
# Usage: docker run <image> -c "make all"
ENTRYPOINT ["/bin/bash"]
CMD ["-c", "make help"]

