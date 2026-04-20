#!/usr/bin/env bash
# Run on your LAPTOP. Packages the repo (excluding secrets + venv + db) and
# ships it to the VM. Safe to run repeatedly — it's an rsync-style overwrite.

set -euo pipefail

VM_HOST="${VM_HOST:-root@178.156.236.211}"
VM_PATH="${VM_PATH:-/opt/pmc}"

cd "$(dirname "$0")/.."

echo "==> Building tarball…"
# Note: we use --exclude with BOTH 'venv' and '.venv' because macOS BSD tar
# and the user's repo can contain either. We also exclude .DS_Store, editor
# junk, and anything that looks like a secret.
# COPYFILE_DISABLE=1 suppresses macOS-specific xattrs that would otherwise
# produce noisy but harmless warnings when Linux tar extracts on the VM.
COPYFILE_DISABLE=1 tar \
    --exclude='.git' \
    --exclude='.github' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='trading.db' \
    --exclude='trading.db-wal' \
    --exclude='trading.db-shm' \
    --exclude='.env' \
    --exclude='.env.local' \
    --exclude='kalshi_private_key.pem' \
    --exclude='*.pem' \
    --exclude='*.key' \
    --exclude='hetzner_key' \
    --exclude='hetzner_key.pub' \
    --exclude='*_key' \
    --exclude='*_key.pub' \
    --exclude='id_rsa*' \
    --exclude='.DS_Store' \
    --exclude='*.swp' \
    --exclude='.idea' \
    --exclude='.vscode' \
    --exclude='data/*.json' \
    --exclude='data/arb_survey.json' \
    --exclude='data/kalshi_series.json' \
    -czf /tmp/pmc.tar.gz .
echo "    → $(du -h /tmp/pmc.tar.gz | awk '{print $1}') tarball ready"

echo "==> Copying to ${VM_HOST}:/tmp/pmc.tar.gz…"
scp /tmp/pmc.tar.gz "${VM_HOST}:/tmp/pmc.tar.gz"

echo "==> Extracting into ${VM_PATH} on the VM…"
# First run: 'pmc' user doesn't exist yet — install.sh creates it.
# Subsequent runs: extract as pmc and restart services.
# We branch on whether the user exists on the VM.
ssh "${VM_HOST}" bash <<EOF
set -e
mkdir -p "${VM_PATH}"
if id pmc >/dev/null 2>&1; then
    echo "    pmc user exists — extracting as pmc and restarting services"
    sudo -u pmc tar -xzf /tmp/pmc.tar.gz -C "${VM_PATH}"
    systemctl restart pmc-llm-bot 2>/dev/null || echo "    (pmc-llm-bot not installed yet)"
    systemctl restart pmc-arb-bot 2>/dev/null || echo "    (pmc-arb-bot not installed yet)"
    systemctl restart pmc-newsalpha-bot 2>/dev/null || echo "    (pmc-newsalpha-bot not installed yet)"
else
    echo "    pmc user not found — first-time install — extracting as root"
    tar -xzf /tmp/pmc.tar.gz -C "${VM_PATH}"
    echo ""
    echo "    NEXT: run the one-time installer:"
    echo "        bash ${VM_PATH}/deploy/install.sh"
fi
EOF
