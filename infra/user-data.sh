#!/bin/bash
set -euo pipefail
export HOME=/root

BASE="https://github.com/parrothacker1/noxos-inference/releases/download/teacher-server-latest"

id -u noxos &>/dev/null || useradd --system --home /opt/noxos-inference --create-home --shell /usr/sbin/nologin noxos

cd /opt/noxos-inference
curl -sfL "${BASE}/teacher-server-linux-amd64" -o teacher-server
curl -sfL "${BASE}/teacher-server-linux-amd64.sha256" -o teacher-server.sha256
echo "$(cut -d' ' -f1 teacher-server.sha256)  teacher-server" | sha256sum -c -
chmod 755 teacher-server
chown noxos:noxos teacher-server

cat > /etc/systemd/system/noxos-inference.service <<'EOF'
[Unit]
Description=noxos-inference teacher-server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=noxos
ExecStart=/opt/noxos-inference/teacher-server
Restart=always
RestartSec=5
Environment=PORT=8443
Environment=NOXOS_MANIFEST_URL=https://github.com/parrothacker1/noxos-inference/releases/download/teacher-latest/manifest.json
Environment=NOXOS_FILE_MANIFEST_URL=https://github.com/parrothacker1/noxos-inference/releases/download/file-latest/manifest.json
Environment=NOXOS_RELOAD_INTERVAL_SECONDS=300
Environment=NOXOS_INFERENCE_API_KEY=CHANGE_ME_BEFORE_LAUNCH

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now noxos-inference
