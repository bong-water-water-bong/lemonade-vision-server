#!/usr/bin/env bash
# launcher for lemonade-vision-server
# When lemond's VisionServer backend starts the server, it invokes this script.
# On first run, it creates a venv and installs dependencies from pyproject.toml.
# Subsequent runs skip the install step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
HOST="${VISION_HOST:-0.0.0.0}"
PORT="${VISION_PORT:-8787}"
DATA_DIR="${VISION_DATA_DIR:-${SCRIPT_DIR}/data}"

if [ ! -d "$VENV_DIR" ]; then
    echo "[vision-server] Creating virtual environment..." >&2
    python3 -m venv "$VENV_DIR"
    echo "[vision-server] Installing dependencies..." >&2
    "$VENV_DIR/bin/pip" install --quiet "$SCRIPT_DIR"
    echo "[vision-server] Dependencies installed." >&2
fi

export VISION_DATA_DIR="$DATA_DIR"

exec "$VENV_DIR/bin/python" -m uvicorn lemonade_vision.server:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT"
