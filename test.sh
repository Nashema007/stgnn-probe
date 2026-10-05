#!/usr/bin/env bash
# Build (if needed) and run commands inside the stgnn-tsl-dev Docker image.
#
# torch_scatter/torch_sparse have no prebuilt wheel for macOS, so this image
# (Linux-based, works identically on macOS host and native Linux) is the
# supported way to run lint/typecheck/tests for this repo. See Dockerfile
# and README.md's "PyTorch Backend Support" section.
#
# Usage:
#   ./test.sh                   # full verification: ruff check, ruff format --check, mypy, pytest
#   ./test.sh pytest tests/ -v  # run an arbitrary command inside the container
#   ./test.sh ruff check src/
#   FORCE_REBUILD=1 ./test.sh   # rebuild the image even if it already exists
#
# On native Linux you can skip Docker entirely — see README.md for the
# `uv sync` + `uv pip install torch-scatter torch-sparse ...` alternative.

set -euo pipefail

IMAGE_NAME="stgnn-tsl-dev"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${FORCE_REBUILD:-0}" == "1" ]] || ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "Building $IMAGE_NAME image..."
    docker build -t "$IMAGE_NAME" "$ROOT_DIR"
fi

if [[ $# -eq 0 ]]; then
    docker run --rm -v "$ROOT_DIR":/app -w /app "$IMAGE_NAME" sh -c \
        "ruff check src/ tests/ scripts/ && ruff format --check src/ tests/ scripts/ && mypy src/ tests/ && pytest tests/ -q"
else
    docker run --rm -v "$ROOT_DIR":/app -w /app "$IMAGE_NAME" "$@"
fi
