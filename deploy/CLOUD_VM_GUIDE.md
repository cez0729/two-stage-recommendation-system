# 云主机选择与部署指南

## 推荐方案：AWS Lightsail x86 Linux

官方定价页：[AWS Lightsail Pricing](https://aws.amazon.com/lightsail/pricing/)

页面当前列出的 Linux/Unix IPv4 套餐中，4 GB 内存、2 vCPU、80 GB SSD 为
24 USD/月；8 GB 内存、2 vCPU、160 GB SSD 为 44 USD/月。这个项目建议先选
4 GB 套餐做 smoke test；如果加载模型和 Grafana 后内存不足，再升到 8 GB。
Lightsail 是固定月费套餐，适合短期展示，使用完应停止或删除实例，避免继续计费。

注册入口：[AWS account sign up](https://aws.amazon.com/resources/create-account/)

创建实例时选择：

1. Linux/Unix，Ubuntu 22.04 LTS 或更新版本。
2. x86_64 架构，2 vCPU，4 GB RAM 起步。
3. 选择离你较近的区域；中国大陆账号和全球 AWS 账号是不同体系，按控制台实际可用区域注册。
4. 防火墙只开放 SSH 22 和 API 8000；Prometheus 9090、Grafana 3000 不要直接暴露公网。

登录并部署：

```bash
ssh ubuntu@<VM_PUBLIC_IP>
sudo apt-get update && sudo apt-get install -y git
git clone <你的代码仓库地址> /opt/recsys
cd /opt/recsys
```

当前本地模型和数据没有提交到 Git，需要从本机额外上传：

```powershell
scp -r .\data\processed ubuntu@<VM_PUBLIC_IP>:/opt/recsys/data/
scp -r .\models ubuntu@<VM_PUBLIC_IP>:/opt/recsys/
scp -r .\artifacts ubuntu@<VM_PUBLIC_IP>:/opt/recsys/
```

然后在 VM 上执行：

```bash
cd /opt/recsys
chmod +x deploy/bootstrap.sh scripts/docker_smoke.sh
./deploy/bootstrap.sh
./scripts/docker_smoke.sh
curl -f http://127.0.0.1:8000/health
```

如果只想临时验证，可以用 SSH 隧道访问本地浏览器，而不是开放监控端口：

```bash
ssh -L 8000:127.0.0.1:8000 -L 3000:127.0.0.1:3000 ubuntu@<VM_PUBLIC_IP>
```

## 免费备选：Oracle Cloud Always Free

官方页面：[Oracle Cloud Free Tier](https://www.oracle.com/cloud/free/)

官方文档当前说明 Always Free 的 Ampere A1 Flex 总量相当于 2 OCPU、12 GB
内存，且需要信用卡验证；资源只能在 home region 创建，可能遇到容量不足。它是
ARM 实例。本项目在本机做了 PyPI 兼容性探测：`faiss-cpu` 有 ARM64 wheel，
但 Python 3.12 ARM64 没有可直接安装的 PyTorch wheel，所以不把它作为首选。
除非你愿意改用 ARM 专用基础镜像或自行编译 PyTorch，否则优先使用 x86 Lightsail。

## 费用和安全

- 先设置云厂商预算告警；不要把银行卡、Access Key、SSH 私钥提交到仓库。
- 测试完成后执行 `docker compose down`，并在云控制台停止或删除实例。
- 本项目没有真实线上流量、CTR、转化率或 GMV 结论；云主机只用于展示可运行的工程闭环。
