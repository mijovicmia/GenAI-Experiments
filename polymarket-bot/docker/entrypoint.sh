#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-paper}"

echo "Starting Polymarket Trading Bot in ${MODE} mode..."

if [ "${MODE}" = "live" ]; then
    exec python scripts/run_live_trading.py
else
    exec python scripts/run_paper_trading.py
fi
