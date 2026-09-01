#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  SUDO=sudo
else
  SUDO=
fi

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  $SUDO apt-get update
  $SUDO apt-get install -y ca-certificates curl
  $SUDO install -m 0755 -d /etc/apt/keyrings
  $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc

  . /etc/os-release
  printf 'Types: deb\nURIs: https://download.docker.com/linux/ubuntu\nSuites: %s\nComponents: stable\nArchitectures: %s\nSigned-By: /etc/apt/keyrings/docker.asc\n' \
    "$VERSION_CODENAME" "$(dpkg --print-architecture)" \
    | $SUDO tee /etc/apt/sources.list.d/docker.sources >/dev/null
  $SUDO apt-get update
  $SUDO apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  $SUDO systemctl enable --now docker
fi

docker compose version >/dev/null
docker compose up -d --build
echo "API: http://127.0.0.1:8000/health"
echo "Prometheus: http://127.0.0.1:9090"
echo "Grafana: http://127.0.0.1:3000"
