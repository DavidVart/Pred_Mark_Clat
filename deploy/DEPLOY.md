# Hetzner Deployment Guide

Target: CPX21 (3 vCPU / 4GB RAM / 80GB disk, us-east Ashburn). Ubuntu 24.04 LTS.

The bot runs as two separate systemd services sharing a single SQLite database:
- `pmc-llm-bot.service` — LLM forecasting bot (paper mode, 60-min cycle)
- `pmc-arb-bot.service` — Cross-platform arbitrage bot (paper mode, 30-sec cycle)

Both are paper-mode by default. Live mode requires editing the unit file AND funding the accounts.

## One-time server setup

SSH into the box as root (or a sudo-capable user) and run:

```bash
# System deps
apt update && apt install -y python3.12 python3.12-venv python3-pip sqlite3 git ufw fail2ban

# Dedicated non-privileged user
useradd -m -s /bin/bash pmc
mkdir -p /opt/pmc /var/log/pmc
chown -R pmc:pmc /opt/pmc /var/log/pmc

# Basic firewall (SSH only; no inbound for the bot)
ufw allow 22/tcp
ufw --force enable
```

## Install the code

From your laptop, push the repo to the VM. Easiest path:

```bash
# On your laptop
cd ~/Desktop/Pred_Mark_Clat
tar --exclude='.git' --exclude='*.db' --exclude='__pycache__' --exclude='.env' \
    -czf /tmp/pmc.tar.gz .
scp /tmp/pmc.tar.gz root@178.156.236.211:/tmp/

# On the VM
sudo -u pmc bash <<'EOF'
cd /opt/pmc
tar -xzf /tmp/pmc.tar.gz
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt
EOF
```

## Secrets

Secrets MUST live outside git. Create `/opt/pmc/.env` on the VM directly (do NOT `scp` a file with secrets in transit if you can avoid it; open a console via Hetzner Cloud if possible):

```bash
sudo -u pmc nano /opt/pmc/.env
# Paste:
# OPENROUTER_API_KEY=sk-or-...
# POLYGON_WALLET_PRIVATE_KEY=0x...
# KALSHI_API_KEY=...
# KALSHI_PRIVATE_KEY_PATH=/opt/pmc/kalshi_private_key.pem
# DAILY_AI_COST_LIMIT=1.00
# SCANNER_MAX_CANDIDATES=3
# MIN_NET_EDGE=0.03
# MAX_POSITIONS_PER_CLUSTER=2
# PREFER_MAKER_ORDERS=true

chmod 600 /opt/pmc/.env
```

Also copy the Kalshi PEM key file over (SCP is OK since it's only the private key for the demo account):

```bash
scp kalshi_private_key.pem root@178.156.236.211:/opt/pmc/kalshi_private_key.pem
ssh root@178.156.236.211 "chown pmc:pmc /opt/pmc/kalshi_private_key.pem && chmod 600 /opt/pmc/kalshi_private_key.pem"
```

## Verify the install

```bash
sudo -u pmc /opt/pmc/venv/bin/python /opt/pmc/cli.py health
```

Expected: all four keys OK, database OK, risk params printed.

## Install the systemd units

```bash
cp /opt/pmc/deploy/systemd/pmc-llm-bot.service /etc/systemd/system/
cp /opt/pmc/deploy/systemd/pmc-arb-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now pmc-llm-bot
systemctl enable --now pmc-arb-bot
```

## Monitor

```bash
# Live status
systemctl status pmc-llm-bot
systemctl status pmc-arb-bot

# Live logs
tail -f /var/log/pmc/llm-bot.log
tail -f /var/log/pmc/arb-bot.log
# or filter to just JSON events:
journalctl -u pmc-llm-bot -f

# Bot status via CLI
sudo -u pmc /opt/pmc/venv/bin/python /opt/pmc/cli.py status
sudo -u pmc /opt/pmc/venv/bin/python /opt/pmc/cli.py arb-status
sudo -u pmc /opt/pmc/venv/bin/python /opt/pmc/cli.py history
```

## Killing the bot remotely

Two mechanisms:

1. **Kill switch via CLI** (preferred — preserves bot state):
   ```bash
   sudo -u pmc /opt/pmc/venv/bin/python /opt/pmc/cli.py kill "reason"
   ```
   Both services keep running but halt all new trades.

2. **Stop the service** (nuclear):
   ```bash
   systemctl stop pmc-llm-bot pmc-arb-bot
   ```

3. **Kill file** — create `/opt/pmc/KILL`; the bot checks for this on every cycle.

## Log rotation

Drop a logrotate config to keep log files from filling the disk:

```bash
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
```

## Updating the bot

```bash
# On laptop
tar --exclude='.git' --exclude='*.db' --exclude='__pycache__' --exclude='.env' \
    -czf /tmp/pmc.tar.gz .
scp /tmp/pmc.tar.gz root@178.156.236.211:/tmp/

# On VM — update + restart
sudo -u pmc tar -xzf /tmp/pmc.tar.gz -C /opt/pmc
sudo -u pmc /opt/pmc/venv/bin/pip install -r /opt/pmc/requirements.txt
systemctl restart pmc-llm-bot pmc-arb-bot
```

(For production, replace tar/scp with a git-based deploy and CI. Out of scope for v1.)

## Going live — checklist

**Don't flip to `--live` until all of these are true:**

- [ ] LLM bot has 50+ resolved paper trades with positive expectancy after fees
- [ ] Platt calibration has fit (check logs for `calibration_fit` event) and shows Brier improvement
- [ ] Arb bot has logged at least 20 `arb_checked` events on real pair data (confirming the matcher finds real markets)
- [ ] You've curated `configs/market_pairs.json` with at least 5 hand-verified pairs
- [ ] Polymarket wallet has USDC on Polygon; Kalshi account is funded
- [ ] `KALSHI_USE_DEMO=false` set in `.env` (currently demo)
- [ ] You're mentally prepared for a single dispute to wipe weeks of grinding
- [ ] Daily loss cap (`MAX_DAILY_LOSS_PCT=0.05` → 5%) set small for live mode
- [ ] You've set up alerting (see below) so you know when things happen

## Monitoring / alerting (recommended before live)

Quick-and-dirty option: a small cron job that pings your phone via Pushover or Telegram when the kill switch activates or the bot dies. Not implemented yet — flag it as a TODO before going live.

## Costs

- Hetzner CPX21: €14.51/mo (~$15)
- OpenRouter LLM: ~$0.80/day at 3 markets/hour × 8 unique/day × $0.04 — capped at $1/day by `DAILY_AI_COST_LIMIT`
- No other recurring costs

**Total fixed run cost: ~$45/mo**. Live trading capital is separate and depends on appetite.
