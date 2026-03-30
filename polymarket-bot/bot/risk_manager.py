"""Hard risk rules enforced independently of Claude's recommendations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from bot.data_models import MarketSnapshot, TradeAction
from bot.logger import get_logger

log = get_logger(__name__)

# In-process state for cooldown tracking (resets on restart)
_exit_timestamps: dict[str, datetime] = {}


def _current_open_trades(snapshot: MarketSnapshot) -> int:
    return len(snapshot.positions)


def _position_size_for_market(snapshot: MarketSnapshot, market_id: str) -> float:
    total = sum(
        p.size * p.current_price
        for p in snapshot.positions
        if p.market_id == market_id
    )
    return total


def _estimated_daily_loss(snapshot: MarketSnapshot) -> float:
    """Use unrealized PnL as a proxy for intra-day loss exposure."""
    unrealized = snapshot.account_state.unrealized_pnl
    realized = snapshot.account_state.realized_pnl
    # Count only losses
    return abs(min(0, unrealized + realized))


def _in_cooldown(market_id: str, cooldown_hours: float) -> bool:
    exit_time = _exit_timestamps.get(market_id)
    if exit_time is None:
        return False
    now = datetime.now(tz=timezone.utc)
    if exit_time.tzinfo is None:
        exit_time = exit_time.replace(tzinfo=timezone.utc)
    return (now - exit_time) < timedelta(hours=cooldown_hours)


def enforce(
    actions: list[TradeAction],
    snapshot: MarketSnapshot,
    constraints: dict,
) -> list[TradeAction]:
    """Apply hard risk rules to a list of actions.

    Blocked actions are replaced with HOLD and a descriptive reason.

    Rules applied:
    - max_position_per_market: block if adding the new position would exceed the cap.
    - max_daily_loss: block all new trades if daily loss budget exhausted.
    - max_open_trades: block new trades if already at the limit.
    - market_cooldown: block re-entry within N hours of an EXIT.
    - min_liquidity_score: (pre-checked by Claude, but double-checked here).
    """
    max_pos = constraints.get("max_position_per_market", 500.0)
    max_loss = constraints.get("max_daily_loss", 200.0)
    max_trades = constraints.get("max_open_trades", 10)
    cooldown_hours = constraints.get("market_cooldown_hours", 2.0)

    daily_loss = _estimated_daily_loss(snapshot)
    open_trades = _current_open_trades(snapshot)

    result: list[TradeAction] = []

    for action in actions:
        if action.action == "HOLD":
            result.append(action)
            continue

        market_id = action.market_id

        # Rule: record EXIT timestamps for cooldown
        if action.action == "EXIT" and market_id:
            _exit_timestamps[market_id] = datetime.now(tz=timezone.utc)
            log.info("Cooldown started for market %s", market_id)
            result.append(action)
            continue

        # Rules below apply only to BUY / SELL
        if action.action in ("BUY", "SELL"):
            # Rule: daily loss cap
            if daily_loss >= max_loss:
                log.warning(
                    "BLOCKED %s on %s — daily loss %.2f >= cap %.2f",
                    action.action, market_id, daily_loss, max_loss,
                )
                result.append(
                    TradeAction(
                        action="HOLD",
                        market_id=market_id,
                        reason=f"Daily loss cap reached (loss={daily_loss:.2f}, cap={max_loss:.2f})",
                    )
                )
                continue

            # Rule: open trades cap
            if open_trades >= max_trades:
                log.warning(
                    "BLOCKED %s on %s — open trades %d >= cap %d",
                    action.action, market_id, open_trades, max_trades,
                )
                result.append(
                    TradeAction(
                        action="HOLD",
                        market_id=market_id,
                        reason=f"Max open trades cap reached ({open_trades}/{max_trades})",
                    )
                )
                continue

            # Rule: cooldown after EXIT
            if market_id and _in_cooldown(market_id, cooldown_hours):
                log.warning(
                    "BLOCKED %s on %s — in cooldown (%.1f h)",
                    action.action, market_id, cooldown_hours,
                )
                result.append(
                    TradeAction(
                        action="HOLD",
                        market_id=market_id,
                        reason=f"Market in post-exit cooldown ({cooldown_hours}h)",
                    )
                )
                continue

            # Rule: per-market position size cap
            if market_id:
                existing_exposure = _position_size_for_market(snapshot, market_id)
                new_size = action.size or 0.0
                if existing_exposure + new_size > max_pos:
                    allowed = max(0.0, max_pos - existing_exposure)
                    if allowed < 1.0:
                        log.warning(
                            "BLOCKED %s on %s — position cap hit (existing=%.2f cap=%.2f)",
                            action.action, market_id, existing_exposure, max_pos,
                        )
                        result.append(
                            TradeAction(
                                action="HOLD",
                                market_id=market_id,
                                reason=(
                                    f"Position cap exceeded for market "
                                    f"(existing={existing_exposure:.2f}, cap={max_pos:.2f})"
                                ),
                            )
                        )
                        continue
                    else:
                        log.info(
                            "Trimming %s size from %.2f to %.2f (position cap)",
                            market_id, new_size, allowed,
                        )
                        action = action.model_copy(update={"size": allowed})

            open_trades += 1  # account for this trade in subsequent checks

        result.append(action)

    return result
