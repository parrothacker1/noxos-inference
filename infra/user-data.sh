#!/bin/bash
# Cloud-init user-data for the noxos-inference box. Ubuntu 22.04/24.04 AMI.
# Installs ClamAV + the FastAPI service, runs it under systemd on boot.
set -euo pipefail
export HOME=/root
exec > /var/log/noxos-inference-setup.log 2>&1

apt-get update -y
apt-get install -y python3-pip python3-venv git clamav-daemon clamav-freshclam

systemctl enable --now clamav-freshclam
systemctl enable --now clamav-daemon

id -u noxos &>/dev/null || useradd --system --no-create-home --shell /usr/sbin/nologin noxos
usermod -aG clamav noxos

git clone https://github.com/parrothacker1/noxos-inference.git /opt/noxos-inference
cd /opt/noxos-inference
python3 -m venv venv
venv/bin/pip install -r requirements.txt
chown -R noxos:noxos /opt/noxos-inference

# Replace before launch (see infra/README.md) — this is a shared-secret bearer
# token, not a real auth system; adequate for a single trusted client app.
API_KEY="CHANGE_ME_BEFORE_LAUNCH"

cat > /etc/systemd/system/noxos-inference.service <<EOF
[Unit]
Description=NoxOS Warden threat-analysis inference API
After=network.target clamav-daemon.service

[Service]
WorkingDirectory=/opt/noxos-inference
Environment=NOXOS_INFERENCE_API_KEY=${API_KEY}
ExecStart=/opt/noxos-inference/venv/bin/uvicorn service.app:app --host 0.0.0.0 --port 8443
Restart=always
RestartSec=5
User=noxos

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now noxos-inference
