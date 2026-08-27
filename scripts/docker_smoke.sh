#!/usr/bin/env bash
set -euo pipefail

KNOWN_USER="${KNOWN_USER:-A0266076X6KPZ6CCHGVS}"
UNKNOWN_USER="${UNKNOWN_USER:-__unknown_smoke_user__}"
COMPOSE=(docker compose)

if ! command -v docker >/dev/null 2>&1; then
  echo "BLOCKED: Docker CLI unavailable" >&2
  exit 2
fi

cleanup() {
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${COMPOSE[@]}" build
"${COMPOSE[@]}" up -d

for attempt in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/tmp/recsys-health.json; then
    break
  fi
  if [ "$attempt" -eq 30 ]; then
    echo "API health check timed out" >&2
    exit 1
  fi
  sleep 2
done

curl -fsS "http://localhost:8000/recommend/${KNOWN_USER}?k=10" >/tmp/recsys-known.json
curl -fsS "http://localhost:8000/recommend/${UNKNOWN_USER}?k=10" >/tmp/recsys-unknown.json
curl -fsS "http://localhost:8000/metrics?format=prometheus" >/tmp/recsys-metrics.txt

grep -q 'request_id' /tmp/recsys-known.json
grep -q 'model_version' /tmp/recsys-known.json
grep -q 'fallback' /tmp/recsys-unknown.json
grep -q 'recsys_request_count' /tmp/recsys-metrics.txt

"${COMPOSE[@]}" stop redis
curl -fsS "http://localhost:8000/recommend/${UNKNOWN_USER}?k=3" >/tmp/recsys-redis-bypass.json
"${COMPOSE[@]}" start redis

echo "Docker smoke checks passed"
