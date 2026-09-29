#!/bin/bash
set -euo pipefail
export HOME=/root

GO_VERSION=1.26.5
GO_TARBALL="go${GO_VERSION}.linux-amd64.tar.gz"
curl -sL "https://go.dev/dl/${GO_TARBALL}" -o "/tmp/${GO_TARBALL}"
rm -rf /usr/local/go
tar -C /usr/local -xzf "/tmp/${GO_TARBALL}"
ln -sf /usr/local/go/bin/go /usr/local/bin/go

id -u noxos &>/dev/null || useradd --system --home /opt/noxos-inference --create-home --shell /usr/sbin/nologin noxos

sudo -u noxos git clone --branch teacher-server --single-branch \
  https://github.com/parrothacker1/noxos-inference.git /opt/noxos-inference/src

cd /opt/noxos-inference/src
sudo -u noxos /usr/local/bin/go build -o /opt/noxos-inference/teacher-server .

cat > /etc/systemd/system/noxos-inference.service <<'EOF'
[Unit]
Description=noxos-inference teacher-server
After=network.target

[Service]
Type=simple
User=noxos
ExecStart=/opt/noxos-inference/teacher-server
Restart=always
RestartSec=5
Environment=PORT=8443
Environment=NOXOS_MANIFEST_URL=https://github.com/parrothacker1/noxos-inference/releases/download/teacher-latest/manifest.json
Environment=NOXOS_RELOAD_INTERVAL_SECONDS=300
Environment=NOXOS_INFERENCE_API_KEY=CHANGE_ME_BEFORE_LAUNCH

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now noxos-inference
