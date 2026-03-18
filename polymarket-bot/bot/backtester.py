"""Historical backtest engine.

Loads JSONL snapshots produced during paper trading (or manually prepared),
steps through time, calls the same decision + risk + execution stack, and
produces performance metrics.

Expected input format (one JSON object per line):
{
  "timestamp": "2024-01-01T00:00:00",
  "markets": [...],          // list of Market dicts
  "orderbooks": {...},       // market_id -> OrderBook dict
  "trades": {...},           // market_id -> [Trade dict]
  "positions": [...],        // (initial state; subsequent positions tracked internally)
  "account_state": {...}
}
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from bot import config, decision_engine, feature_engineering, risk_manager
from bot.data_models import (
    AccountState,
    EngineeredMarket,
    Market,
    MarketSnapshot,
    Order,
    OrderBook,
    Position,
    PriceLevel,
    Trade,
    TradeAction,
)
from bot.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Snapshot loading
# ---------------------------------------------------------------------------


def _parse_orderbook(market_id: str, data: dict) -> OrderBook:
    return OrderBook(
        market_id=market_id,
        token_id=data.get("token_id", ""),
        bids=[PriceLevel(**b) for b in data.get("bids", [])],
        asks=[PriceLevel(**a) for a in data.get("asks", [])],
        timestamp=datetime.fromisoformat(
            data.get("timestamp", datetime.utcnow().isoformat())
        ),
    )


def _parse_trade(market_id: str, data: dict) -> Trade:
    return Trade(
        trade_id=data.get("trade_id", ""),
        market_id=market_id,
        timestamp=datetime.fromisoformat(data["timestamp"]),
        price=float(data["price"]),
        size=float(data["size"]),
        side=data.get("side", "BUY"),
    )


def load_snapshots(path: Path) -> Iterator[MarketSnapshot]:
    """Yield MarketSnapshot objects from a JSONL file."""
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", line_no, exc)
                continue

            markets = [Market(**m) for m in raw.get("markets", [])]
            orderbooks = {
                mid: _parse_orderbook(mid, ob)
                for mid, ob in raw.get("orderbooks", {}).items()
            }
            trades = {
                mid: [_parse_trade(mid, t) for t in tlist]
                for mid, tlist in raw.get("trades", {}).items()
            }
            positions: list[Position] = []  # managed internally during backtest
            account_state = AccountState(**raw.get("account_state", {"cash": 10_000.0}))

            yield MarketSnapshot(
                markets=markets,
                orderbooks=orderbooks,
                trades=trades,
                positions=positions,
                account_state=account_state,
                timestamp=datetime.fromisoformat(raw["timestamp"]),
            )


# ---------------------------------------------------------------------------
# Simulated position tracking
# ---------------------------------------------------------------------------


class BacktestPortfolio:
    """Tracks positions and PnL during a backtest run."""

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}  # market_id -> {side, size, entry_price}
        self.realized_pnl = 0.0
        self.equity_curve: list[dict] = []

    def apply_action(self, action: TradeAction, snapshot: MarketSnapshot) -> None:
        market_id = action.market_id
        if not market_id or action.action == "HOLD":
            return

        ob = snapshot.orderbooks.get(market_id)
        current_price = ob.mid_price if ob and ob.mid_price else (action.limit_price or 0.5)

        if action.action == "BUY":
            cost = (action.size or 0.0)
            if cost > self.cash:
                log.warning("Insufficient cash for BUY on %s — skipping", market_id)
                return
            self.cash -= cost
            pos = self.positions.get(market_id, {"side": action.side, "size": 0.0, "entry_price": 0.0})
            old_size = pos["size"]
            new_size = old_size + (action.size or 0.0) / (action.limit_price or 0.5)
            old_entry = pos["entry_price"]
            if new_size > 0:
                new_entry = ((old_size * old_entry) + ((action.size or 0.0) / (action.limit_price or 0.5) * (action.limit_price or 0.5))) / new_size
            else:
                new_entry = action.limit_price or 0.5
            self.positions[market_id] = {"side": action.side, "size": new_size, "entry_price": new_entry}

        elif action.action in ("SELL", "EXIT"):
            pos = self.positions.get(market_id)
            if not pos:
                return
            sell_size = (action.size or pos["size"]) if action.action == "SELL" else pos["size"]
            sell_size = min(sell_size, pos["size"])
            proceeds = sell_size * current_price
            self.cash += proceeds
            pnl = (current_price - pos["entry_price"]) * sell_size
            self.realized_pnl += pnl
            if sell_size >= pos["size"]:
                del self.positions[market_id]
            else:
                self.positions[market_id]["size"] -= sell_size

    def record_equity(self, ts: datetime, snapshot: MarketSnapshot) -> None:
        unrealized = 0.0
        for market_id, pos in self.positions.items():
            ob = snapshot.orderbooks.get(market_id)
            price = ob.mid_price if ob and ob.mid_price else pos["entry_price"]
            unrealized += (price - pos["entry_price"]) * pos["size"]

        equity = self.cash + sum(
            snapshot.orderbooks[mid].mid_price * pos["size"]
            for mid, pos in self.positions.items()
            if mid in snapshot.orderbooks and snapshot.orderbooks[mid].mid_price
        )
        self.equity_curve.append(
            {
                "timestamp": ts.isoformat(),
                "cash": self.cash,
                "unrealized_pnl": unrealized,
                "realized_pnl": self.realized_pnl,
                "equity": equity,
            }
        )

    def as_positions_list(self, snapshot: MarketSnapshot) -> list[Position]:
        out: list[Position] = []
        for market_id, pos in self.positions.items():
            ob = snapshot.orderbooks.get(market_id)
            cur_price = ob.mid_price if ob and ob.mid_price else pos["entry_price"]
            out.append(
                Position(
                    market_id=market_id,
                    token_id="",
                    side=pos["side"] or "YES",
                    size=pos["size"],
                    entry_price=pos["entry_price"],
                    current_price=cur_price,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(portfolio: BacktestPortfolio, initial_cash: float) -> dict:
    curve = [e["equity"] for e in portfolio.equity_curve]
    if not curve:
        return {}

    peak = curve[0]
    max_dd = 0.0
    for equity in curve:
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak else 0
        max_dd = max(max_dd, dd)

    total_return = (curve[-1] - initial_cash) / initial_cash if initial_cash else 0
    wins = [e for e in portfolio.equity_curve if e["realized_pnl"] > 0]
    losses = [e for e in portfolio.equity_curve if e["realized_pnl"] < 0]

    return {
        "initial_cash": initial_cash,
        "final_equity": curve[-1],
        "total_return_pct": round(total_return * 100, 2),
        "realized_pnl": round(portfolio.realized_pnl, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "cycles_run": len(curve),
    }


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    data_path: Path,
    output_dir: Path,
    initial_cash: float = 10_000.0,
    constraints: dict | None = None,
) -> dict:
    """Run a full backtest over a JSONL snapshot file.

    Args:
        data_path: Path to JSONL file with historical snapshots.
        output_dir: Directory where equity curve CSV will be written.
        initial_cash: Starting USDC balance.
        constraints: Risk constraints dict (defaults to config values).

    Returns:
        Dict of summary metrics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints = constraints or config.load_constraints()

    portfolio = BacktestPortfolio(initial_cash=initial_cash)
    cycle = 0

    for snapshot in load_snapshots(data_path):
        cycle += 1
        log.info("Backtest cycle %d — %s", cycle, snapshot.timestamp.isoformat())

        # Inject tracked positions into snapshot
        snapshot.positions = portfolio.as_positions_list(snapshot)
        snapshot.account_state = AccountState(
            cash=portfolio.cash,
            realized_pnl=portfolio.realized_pnl,
            unrealized_pnl=sum(
                (snapshot.orderbooks[mid].mid_price or pos.entry_price) * pos.size
                - pos.entry_price * pos.size
                for mid, pos_dict in portfolio.positions.items()
                for pos in [Position(
                    market_id=mid, token_id="", side=pos_dict["side"] or "YES",
                    size=pos_dict["size"], entry_price=pos_dict["entry_price"],
                    current_price=(snapshot.orderbooks[mid].mid_price if mid in snapshot.orderbooks and snapshot.orderbooks[mid].mid_price else pos_dict["entry_price"]),
                )]
            ),
        )

        engineered = feature_engineering.build_features(snapshot)
        if not engineered:
            continue

        actions = decision_engine.decide(snapshot, engineered, constraints)
        final_actions = risk_manager.enforce(actions, snapshot, constraints)

        for action in final_actions:
            portfolio.apply_action(action, snapshot)

        portfolio.record_equity(snapshot.timestamp, snapshot)

    # Write equity curve CSV
    equity_path = output_dir / "equity_curve.csv"
    if portfolio.equity_curve:
        with open(equity_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=portfolio.equity_curve[0].keys())
            writer.writeheader()
            writer.writerows(portfolio.equity_curve)
        log.info("Equity curve written to %s", equity_path)

    metrics = compute_metrics(portfolio, initial_cash)
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics written to %s", metrics_path)

    return metrics
