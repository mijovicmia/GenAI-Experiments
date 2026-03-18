"""Decision engine — builds SkillInput, calls Claude, and filters results."""

from __future__ import annotations

from bot import claude_client
from bot.data_models import (
    EngineeredMarket,
    MarketSnapshot,
    SkillInput,
    SkillInputMarket,
    SkillInputPosition,
    TradeAction,
)
from bot.logger import get_logger

log = get_logger(__name__)


def decide(
    snapshot: MarketSnapshot,
    engineered_markets: list[EngineeredMarket],
    constraints: dict,
) -> list[TradeAction]:
    """Build SkillInput, query Claude, and return filtered TradeActions.

    Steps:
    1. Assemble the SkillInput from snapshot + engineered features.
    2. Call claude_client.get_trade_actions.
    3. Filter out actions with obvious problems (unknown markets, negative size, etc.).
    """
    # Step 1: build SkillInput
    known_market_ids = {em.market_id for em in engineered_markets}

    skill_markets = [
        SkillInputMarket(
            market_id=em.market_id,
            name=em.name,
            implied_prob=em.implied_prob,
            spread_bps=em.spread_bps,
            momentum_score=em.momentum_score,
            liquidity_score=em.liquidity_score,
            time_to_expiry_hours=em.time_to_expiry_hours,
        )
        for em in engineered_markets
    ]

    skill_positions = [
        SkillInputPosition(
            market_id=pos.market_id,
            side=pos.side,
            size=pos.size,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            unrealized_pnl=pos.unrealized_pnl,
        )
        for pos in snapshot.positions
    ]

    skill_input = SkillInput(
        timestamp=snapshot.timestamp.isoformat(),
        markets=skill_markets,
        positions=skill_positions,
        constraints=constraints,
    )

    log.info(
        "Calling Claude with %d markets and %d positions",
        len(skill_markets),
        len(skill_positions),
    )

    # Step 2: call Claude
    raw_actions = claude_client.get_trade_actions(skill_input.model_dump())

    # Step 3: filter
    filtered: list[TradeAction] = []
    for action in raw_actions:
        if action.action == "HOLD":
            filtered.append(action)
            continue

        # Must reference a known market
        if action.market_id not in known_market_ids:
            log.warning(
                "Dropping action for unknown market_id=%s", action.market_id
            )
            continue

        # Size must be positive
        if action.size is not None and action.size <= 0:
            log.warning(
                "Dropping action with non-positive size=%.4f for %s",
                action.size, action.market_id,
            )
            continue

        # limit_price must be in (0, 1)
        if action.limit_price is not None and not (0 < action.limit_price < 1):
            log.warning(
                "Dropping action with out-of-range limit_price=%.4f for %s",
                action.limit_price, action.market_id,
            )
            continue

        filtered.append(action)

    if not filtered:
        log.info("All actions filtered out — inserting HOLD")
        filtered.append(
            TradeAction(
                action="HOLD",
                market_id=None,
                reason="All Claude actions were filtered by decision engine",
            )
        )

    log.info(
        "Decision engine returning %d action(s): %s",
        len(filtered),
        [(a.action, a.market_id) for a in filtered],
    )
    return filtered
