#!/usr/bin/env bash
set -Eeuo pipefail

REGION="ap-northeast-2"
INSTANCE="recsys-demo"
HOST="${RECSYS_HOST:?Set RECSYS_HOST to the VM public IP}"
KEY="${1:?Pass the SSH private-key path as argument 1}"
BUNDLE="${2:-recsys-deploy.tar.gz}"

cleanup_local() {
  aws lightsail close-instance-public-ports \
    --region "$REGION" \
    --instance-name "$INSTANCE" \
    --port-info fromPort=443,toPort=443,protocol=tcp \
    >/dev/null 2>&1 || true
  rm -f -- "$KEY"
}
trap cleanup_local EXIT

test -f "$KEY" || { echo "Missing SSH key: $KEY" >&2; exit 2; }
test -f "$BUNDLE" || { echo "Missing deployment bundle: $BUNDLE" >&2; exit 2; }
chmod 600 "$KEY"

SSH=(ssh -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
  -o ServerAliveInterval=30 "ubuntu@$HOST")
SCP=(scp -i "$KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)

echo "[1/6] Opening the public API port"
aws lightsail open-instance-public-ports \
  --region "$REGION" \
  --instance-name "$INSTANCE" \
  --port-info fromPort=8000,toPort=8000,protocol=tcp,cidrs=0.0.0.0/0 \
  >/dev/null

echo "[2/6] Uploading the deployment bundle"
"${SCP[@]}" "$BUNDLE" "ubuntu@$HOST:/tmp/recsys-deploy.tar.gz"

echo "[3/6] Installing and starting the recommendation stack"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -Eeuo pipefail

on_error() {
  if [[ -f /opt/recsys/docker-compose.yml ]]; then
    cd /opt/recsys
    sudo docker compose ps || true
    sudo docker compose logs --tail=120 || true
  fi
}
trap on_error ERR

if [[ ! -f /swapfile ]]; then
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
fi
if ! sudo swapon --show=NAME --noheadings | grep -qx '/swapfile'; then
  sudo swapon /swapfile
fi
grep -q '^/swapfile ' /etc/fstab \
  || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null

sudo install -d -o ubuntu -g ubuntu /opt/recsys
sudo tar -xzf /tmp/recsys-deploy.tar.gz -C /opt/recsys
sudo chown -R ubuntu:ubuntu /opt/recsys
cd /opt/recsys
sudo bash deploy/bootstrap.sh
sudo bash scripts/docker_smoke.sh
sudo docker compose up -d

for attempt in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8000/health >/tmp/recsys-health.json; then
    break
  fi
  [[ "$attempt" -lt 30 ]] || { echo 'Final health check timed out' >&2; exit 1; }
  sleep 2
done

curl -fsS 'http://127.0.0.1:8000/recommend/A0266076X6KPZ6CCHGVS?k=10' \
  >/tmp/recsys-known.json
curl -fsS 'http://127.0.0.1:8000/recommend/__cloud_unknown__?k=5' \
  >/tmp/recsys-unknown.json
grep -q 'recsys_baseline_v1' /tmp/recsys-health.json
grep -q 'request_id' /tmp/recsys-known.json
grep -q 'fallback' /tmp/recsys-unknown.json

sudo docker compose ps
cat /tmp/recsys-health.json
echo
echo 'REMOTE_DEPLOYMENT_OK'

sudo rm -f /tmp/recsys-deploy.tar.gz
sudo rm -f /etc/ssh/sshd_config.d/99-recsys-upload.conf
if [[ -f /run/sshd-443.pid ]]; then
  sudo kill "$(cat /run/sshd-443.pid)" 2>/dev/null || true
  sudo rm -f /run/sshd-443.pid
fi
REMOTE

echo "[4/6] Closing the temporary SSH port"
aws lightsail close-instance-public-ports \
  --region "$REGION" \
  --instance-name "$INSTANCE" \
  --port-info fromPort=443,toPort=443,protocol=tcp \
  >/dev/null || true

echo "[5/6] Checking the public endpoint"
curl -fsS --retry 10 --retry-delay 3 "http://$HOST:8000/health"
echo

echo "[6/6] Deployment complete"
echo "API health: http://$HOST:8000/health"
echo "API docs:   http://$HOST:8000/docs"
echo "CLOUD_DEPLOYMENT_OK"
