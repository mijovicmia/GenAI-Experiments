#!/usr/bin/env python3
"""CLI for running historical backtests.

Usage:
    python scripts/run_backtest.py --data data/snapshots.jsonl --output data/backtest_results
    python scripts/run_backtest.py --data data/snapshots.jsonl --cash 5000
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.backtester import run_backtest
from bot.logger import get_logger

log = get_logger("backtest")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Polymarket trading bot backtest")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to JSONL file containing historical market snapshots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/backtest_results"),
        help="Directory to write equity curve CSV and metrics JSON",
    )
    parser.add_argument(
        "--cash",
        type=float,
        default=10_000.0,
        help="Starting USDC balance for the backtest",
    )
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Data file not found: %s", args.data)
        sys.exit(1)

    log.info("Starting backtest: data=%s, output=%s, initial_cash=%.2f", args.data, args.output, args.cash)
    metrics = run_backtest(
        data_path=args.data,
        output_dir=args.output,
        initial_cash=args.cash,
    )

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(json.dumps(metrics, indent=2))
    print(f"\nEquity curve: {args.output / 'equity_curve.csv'}")
    print(f"Metrics JSON: {args.output / 'metrics.json'}")


if __name__ == "__main__":
    main()
