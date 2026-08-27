#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  SUDO=sudo
else
  SUDO=
fi

if ! command -v docker >/dev/null 2>&1; then
  $SUDO apt-get update
  $SUDO apt-get install -y ca-certificates curl docker.io docker-compose-plugin
  $SUDO systemctl enable --now docker
fi

docker compose version >/dev/null
docker compose up -d --build
echo "API: http://127.0.0.1:8000/health"
echo "Prometheus: http://127.0.0.1:9090"
echo "Grafana: http://127.0.0.1:3000"
