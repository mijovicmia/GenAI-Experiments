"""Order executor — paper-trades or sends live orders to the exchange."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from bot.data_models import TradeAction
from bot.exchange_client import ExchangeClient, ExchangeClientError
from bot.logger import get_logger

log = get_logger(__name__)

_PAPER_LOG_DIR = Path("logs")
_PAPER_TRADES_CSV = _PAPER_LOG_DIR / "paper_trades.csv"
_PAPER_TRADES_JSONL = _PAPER_LOG_DIR / "paper_trades.jsonl"

# In-memory paper positions: market_id -> {"side": str, "size": float, "entry_price": float}
_paper_positions: dict[str, dict] = {}


def _ensure_paper_log() -> None:
    _PAPER_LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not _PAPER_TRADES_CSV.exists():
        with open(_PAPER_TRADES_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "action", "market_id", "side", "size", "limit_price", "reason"]
            )


def _log_paper_trade(action: TradeAction) -> None:
    _ensure_paper_log()
    ts = datetime.now(tz=timezone.utc).isoformat()

    with open(_PAPER_TRADES_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ts,
                action.action,
                action.market_id,
                action.side,
                action.size,
                action.limit_price,
                action.reason,
            ]
        )

    with open(_PAPER_TRADES_JSONL, "a") as f:
        record = action.model_dump()
        record["timestamp"] = ts
        f.write(json.dumps(record) + "\n")


def _simulate_paper_position(action: TradeAction) -> None:
    """Update in-memory paper positions based on the action."""
    market_id = action.market_id
    if not market_id:
        return

    if action.action in ("BUY",):
        existing = _paper_positions.get(market_id, {"size": 0.0, "entry_price": 0.0})
        old_size = existing["size"]
        new_size = old_size + (action.size or 0.0)
        old_entry = existing.get("entry_price", action.limit_price or 0.5)
        if new_size > 0:
            new_entry = (
                (old_size * old_entry) + ((action.size or 0.0) * (action.limit_price or 0.5))
            ) / new_size
        else:
            new_entry = action.limit_price or 0.5
        _paper_positions[market_id] = {
            "side": action.side,
            "size": new_size,
            "entry_price": new_entry,
        }
        log.info(
            "[PAPER] BUY %s | side=%s size=%.2f @ %.4f | new_total=%.2f",
            market_id, action.side, action.size, action.limit_price or 0, new_size,
        )

    elif action.action in ("SELL", "EXIT"):
        if market_id in _paper_positions:
            pos = _paper_positions[market_id]
            if action.action == "EXIT" or (action.size or 0) >= pos["size"]:
                del _paper_positions[market_id]
                log.info("[PAPER] EXIT %s — position closed", market_id)
            else:
                pos["size"] -= action.size or 0.0
                log.info(
                    "[PAPER] SELL %s | size=%.2f remaining=%.2f",
                    market_id, action.size, pos["size"],
                )
        else:
            log.warning("[PAPER] SELL/EXIT for %s — no open position found", market_id)


def execute(
    actions: list[TradeAction],
    mode: str,
    client: ExchangeClient,
    token_id_map: dict[str, str] | None = None,
) -> None:
    """Execute a list of trade actions.

    Args:
        actions: Validated, risk-checked actions.
        mode: "paper" or "live".
        client: Initialised ExchangeClient.
        token_id_map: Optional mapping from market_id -> YES token_id (needed for live).
    """
    token_id_map = token_id_map or {}

    for action in actions:
        if action.action == "HOLD":
            log.debug("HOLD on %s — %s", action.market_id, action.reason)
            continue

        if mode == "paper":
            _log_paper_trade(action)
            _simulate_paper_position(action)
        elif mode == "live":
            _execute_live(action, client, token_id_map)
        else:
            log.error("Unknown mode '%s' — skipping execution", mode)


def _execute_live(
    action: TradeAction,
    client: ExchangeClient,
    token_id_map: dict[str, str],
) -> None:
    """Send a single action to the exchange."""
    market_id = action.market_id
    if not market_id:
        log.warning("Skipping action with no market_id: %s", action)
        return

    token_id = token_id_map.get(market_id)
    if not token_id:
        log.error("No token_id found for market %s — cannot place order", market_id)
        return

    if action.action == "EXIT":
        # EXIT: place a SELL order at market (use best bid as limit)
        clob_side = "SELL"
        size = action.size or 0.0
        price = action.limit_price or 0.01
    elif action.action == "SELL":
        clob_side = "SELL"
        size = action.size or 0.0
        price = action.limit_price or 0.01
    else:  # BUY
        clob_side = "BUY"
        size = action.size or 0.0
        price = action.limit_price or 0.99

    if size <= 0:
        log.warning("Skipping zero-size order for %s", market_id)
        return

    try:
        order = client.place_order(
            token_id=token_id,
            side=clob_side,
            size=size,
            price=price,
            market_id=market_id,
        )
        log.info(
            "[LIVE] Order placed: id=%s market=%s side=%s size=%.4f price=%.4f",
            order.order_id, market_id, clob_side, size, price,
        )
    except ExchangeClientError as exc:
        log.error("Order placement failed for %s: %s", market_id, exc)


def get_paper_positions() -> dict[str, dict]:
    """Return current in-memory paper positions (for backtesting / status checks)."""
    return dict(_paper_positions)
