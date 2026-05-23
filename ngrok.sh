#!/usr/bin/env bash
# Start ngrok tunnel for lemonade-vision-server on :8787
# Usage: ./ngrok.sh [authtoken]
set -e

PORT=8787

if [ -n "$1" ]; then
  ngrok config add-authtoken "$1"
fi

echo "Starting ngrok tunnel → localhost:${PORT}"
ngrok http ${PORT}
