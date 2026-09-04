# OpenFinGym RL training image.
#
# Layout:
#   /opt/venv   - the python venv with skyrl[fsdp] + harbor + RL-only OpenFinGym deps
#   /workspace  - mount point for the code PVC (OpenFinGym1 + SkyRL-main side by side)
#
# The image contains the ENVIRONMENT (heavy deps). The CODE comes in via PVC mount
# at runtime, exposed to python via PYTHONPATH — no pip install -e at Pod start.
# This means you rebuild the image only when deps change, and `git pull` in the PVC
# is enough to pick up code changes.
#
# Build:
#   cd deploy/docker
#   bash build_and_push.sh <registry/repo:tag>
#
# Approx final size: ~12 GB compressed / ~25 GB uncompressed.

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# System dependencies
# - python 3.12 (the interpreter version the project targets; see pyproject.toml)
# - apptainer (for the in-Pod Singularity sandbox provider)
# - build tools (some pip wheels need them; flash-attn comes from a prebuilt wheel
#   so no nvcc compile needed, but build-essential is small enough to keep)
# - git (for cloning SkyRL into the image and for in-Pod git pull)
# - curl/ca-certificates (for uv installer and HF model downloads)

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && add-apt-repository -y ppa:apptainer/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        python3-pip \
        apptainer \
        git curl ca-certificates \
        build-essential \
        rsync \
        less vim-tiny \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python \
 && ln -sf /usr/bin/python3.12 /usr/local/bin/python3

# uv installer
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
 && mv /root/.local/bin/uv /usr/local/bin/uv \
 && mv /root/.local/bin/uvx /usr/local/bin/uvx

# Python environment
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"
RUN python -m venv "${VIRTUAL_ENV}" \
 && uv pip install --python "${VIRTUAL_ENV}/bin/python" --upgrade pip setuptools wheel

# SkyRL with FSDP
# Pulls torch+cu128, vllm+cu129, flash-attn (prebuilt wheel), ray, accelerate, etc.
# The fsdp extra is what `run_codecontest.sh` uses upstream.
#
# SkyRL revision to install. Defaults to `main`; override with a branch, tag
# or commit SHA via `--build-arg SKYRL_REV=<sha>` to pin the image against the
# SkyRL-main checkout on the PVC (bump when upstream changes the
# HarborGenerator API).
ARG SKYRL_REV=main
RUN git clone https://github.com/NovaSky-AI/SkyRL /tmp/skyrl \
 && git -C /tmp/skyrl checkout --quiet "${SKYRL_REV}" \
 && uv pip install --python "${VIRTUAL_ENV}/bin/python" "/tmp/skyrl[fsdp]" \
 && rm -rf /tmp/skyrl

# Harbor compatibility extras
# Pinned commit matches SkyRL's own pin (SkyRL-main/pyproject.toml).
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
    "harbor[daytona,modal] @ git+https://github.com/laude-institute/harbor@a85628c803dc1713fcd26d2bb1908cde69d62317"

# OpenFinGym RL dependencies
# A curated subset of the OpenFinGym pyproject.toml — only the deps the agent +
# verifier need at training time. Auto-pipeline deps (langchain*, ccxt, akshare,
# kaggle, etc.) are intentionally NOT installed; openfinai_pipeline is never
# imported during an RL run.
RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" \
    "pydantic>=2.12.5,<3.0.0" \
    "omegaconf>=2.3.0,<3.0.0" \
    "python-dotenv>=1.0.1,<2.0.0" \
    "requests>=2.33.1,<3.0.0" \
    "fastapi<1.0.0" \
    "uvicorn[standard]<1.0.0" \
    "filelock<4.0.0" \
    "jinja2>=3.1.6,<4.0.0" \
    "huggingface_hub>=0.26.0,<1.0.0" \
    "litellm<2.0" \
    "tomli>=2.0" \
    "tomli-w>=1.0" \
    "loguru" \
    "h5py<4.0" \
    "pyarrow<25.0" \
    "scipy>=1.17.1,<2.0.0" \
    "wandb"

# Sanity check
RUN python -c "import torch, skyrl, harbor, litellm; \
    print('torch=', torch.__version__, 'cuda=', torch.version.cuda); \
    import vllm; print('vllm=', vllm.__version__); \
    import flash_attn; print('flash_attn=', flash_attn.__version__)"

# Default runtime
# Code (OpenFinGym1 + SkyRL-main) is mounted at /workspace by the K8s manifest.
# PYTHONPATH is set by the Job's command — keeping this layer image-side keeps
# things flexible.
WORKDIR /workspace

CMD ["/bin/bash"]
