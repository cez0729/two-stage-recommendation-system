# Single-VM Deployment

This is an executable Docker Compose deployment asset for one Ubuntu VM (AWS EC2 or
GCP Compute Engine). It is not a claim that a remote instance has been deployed.

## Prerequisites

- Ubuntu 22.04+ VM with at least 4 GB RAM and a security group/firewall allowing TCP 8000.
- The repository checkout plus the local `data/`, `models/`, and `artifacts/` directories.
- No cloud credentials are stored in this repository.

## Bootstrap

```bash
cd /path/to/推荐系统
chmod +x deploy/bootstrap.sh
./deploy/bootstrap.sh
```

The script installs Docker Engine and the Compose plugin when absent, then builds and
starts API, Redis, Prometheus, and Grafana. It prints local health URLs after startup.

## Verify and stop

```bash
./scripts/docker_smoke.sh
docker compose ps
docker compose down
```

Endpoints are `/health`, `/recommend/<user_id>`, `/events`, and `/metrics?format=prometheus`.
Prometheus is on port 9090 and Grafana is on port 3000. Put an authenticated reverse
proxy in front of these ports before exposing them to the public internet.

## Operational limits

The current benchmark is offline/local: P50 175.19 ms, P95 191.49 ms, 5.68 QPS. No
production traffic, online A/B lift, CTR, conversion, or GMV claim is made. Docker
deployment must be verified on the target VM and recorded separately.
