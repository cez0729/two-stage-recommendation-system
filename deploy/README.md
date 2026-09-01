# Single-VM Deployment

This is an executable Docker Compose deployment asset for one Ubuntu VM (AWS EC2 or
GCP Compute Engine). It is not a claim that a remote instance has been deployed.

For provider selection, pricing, asset upload, and SSH tunnel commands, see
[`CLOUD_VM_GUIDE.md`](CLOUD_VM_GUIDE.md). The recommended first target is an x86 AWS
Lightsail Linux instance because the repository depends on PyTorch and FAISS.

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

The 2026-09-01 local rerun measured P50 26.59 ms, P95 30.48 ms, and 37.24 QPS with
the 200-candidate pool unchanged. A 20-user direct profile measured P95 31.69 ms and
identified feature construction as 57.64% of mean pipeline latency. These are local
measurements, not a production SLA. No production traffic, online A/B lift, CTR,
conversion, or GMV claim is made. Docker deployment must be verified on the target VM
and recorded separately.
