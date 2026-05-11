# Shared base: Python 3.11 + CPU-only PyTorch pre-installed
# Build once with:  docker compose build base
# All ML services inherit from this to avoid re-downloading torch.
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc g++ sqlite3 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# CPU-only torch – installed first so that no downstream package
# can upgrade it to a GPU wheel.
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

