#!/usr/bin/env bash
# One-shot first-time install. Run as ROOT on the Hetzner VM.
#
#   ssh root@178.156.236.211
#   bash /opt/pmc/deploy/install.sh
#
# Idempotent — safe to re-run. Skips steps that are already done.

set -euo pipefail

PMC_USER="pmc"
PMC_DIR="/opt/pmc"
LOG_DIR="/var/log/pmc"

echo "==> System packages…"
apt-get update -qq
apt-get install -y -qq \
    python3.12 python3.12-venv python3-pip \
    sqlite3 git curl ufw logrotate

echo "==> Non-privileged user '${PMC_USER}'…"
if ! id "${PMC_USER}" >/dev/null 2>&1; then
    useradd -m -s /bin/bash "${PMC_USER}"
fi
mkdir -p "${PMC_DIR}" "${LOG_DIR}"
chown -R "${PMC_USER}:${PMC_USER}" "${PMC_DIR}" "${LOG_DIR}"

echo "==> Firewall (SSH-only)…"
ufw allow 22/tcp >/dev/null
yes | ufw enable >/dev/null || true

echo "==> Python venv…"
if [ ! -d "${PMC_DIR}/venv" ]; then
    sudo -u "${PMC_USER}" python3.12 -m venv "${PMC_DIR}/venv"
fi
sudo -u "${PMC_USER}" "${PMC_DIR}/venv/bin/pip" install --quiet --upgrade pip
sudo -u "${PMC_USER}" "${PMC_DIR}/venv/bin/pip" install --quiet -r "${PMC_DIR}/requirements.txt"

echo "==> Systemd units…"
cp "${PMC_DIR}/deploy/systemd/pmc-llm-bot.service" /etc/systemd/system/
cp "${PMC_DIR}/deploy/systemd/pmc-arb-bot.service" /etc/systemd/system/
systemctl daemon-reload

echo "==> Logrotate…"
cat > /etc/logrotate.d/pmc <<'EOF'
/var/log/pmc/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    su pmc pmc
}
EOF

echo
echo "==> Install complete. Next:"
echo
echo "  1. Create /opt/pmc/.env (secrets — do NOT scp):"
echo "     sudo -u ${PMC_USER} nano ${PMC_DIR}/.env"
echo
echo "  2. Copy your Kalshi private key:"
echo "     # From your laptop:"
echo "     scp kalshi_private_key.pem root@<VM_IP>:${PMC_DIR}/kalshi_private_key.pem"
echo "     # Then on VM:"
echo "     chown ${PMC_USER}:${PMC_USER} ${PMC_DIR}/kalshi_private_key.pem"
echo "     chmod 600 ${PMC_DIR}/kalshi_private_key.pem"
echo
echo "  3. Verify:"
echo "     sudo -u ${PMC_USER} ${PMC_DIR}/venv/bin/python ${PMC_DIR}/cli.py health"
echo
echo "  4. Start services:"
echo "     systemctl enable --now pmc-llm-bot"
echo "     systemctl enable --now pmc-arb-bot   # will idle until pairs are seeded"
echo
echo "  5. Monitor:"
echo "     tail -f ${LOG_DIR}/llm-bot.log"
