"""Main run loop — orchestrates the full trading cycle."""

from __future__ import annotations

import time

from bot import config, data_fetcher, decision_engine, feature_engineering, risk_manager
from bot.exchange_client import ExchangeClient
from bot.logger import get_logger
from bot.order_executor import execute

log = get_logger(__name__)


def _build_token_id_map(snapshot) -> dict[str, str]:
    """Build a {market_id: yes_token_id} mapping from the snapshot."""
    mapping: dict[str, str] = {}
    for market in snapshot.markets:
        if market.outcomes:
            mapping[market.id] = market.outcomes[0].token_id
    return mapping


def run(mode: str = config.MODE) -> None:
    """Run the trading bot loop indefinitely.

    Args:
        mode: "paper" or "live".
    """
    config.validate()
    log.info("=" * 60)
    log.info("Polymarket Claude Bot starting in %s mode", mode.upper())
    log.info("Poll interval: %ds", config.POLL_INTERVAL_SECONDS)
    log.info("=" * 60)

    client = ExchangeClient()
    constraints = config.load_constraints()

    cycle = 0
    while True:
        cycle += 1
        log.info("--- Cycle %d ---", cycle)

        try:
            # 1. Fetch market data
            snapshot = data_fetcher.build_snapshot(client)

            # 2. Compute features
            engineered = feature_engineering.build_features(snapshot)

            if not engineered:
                log.warning("No engineered markets available — skipping cycle")
                time.sleep(config.POLL_INTERVAL_SECONDS)
                continue

            # 3. Get decisions from Claude
            actions = decision_engine.decide(snapshot, engineered, constraints)

            # 4. Apply hard risk rules
            final_actions = risk_manager.enforce(actions, snapshot, constraints)

            # 5. Execute (paper or live)
            token_map = _build_token_id_map(snapshot)
            execute(final_actions, mode=mode, client=client, token_id_map=token_map)

        except KeyboardInterrupt:
            log.info("Shutdown requested — exiting loop")
            break
        except Exception as exc:
            log.exception("Unhandled error in cycle %d: %s", cycle, exc)

        log.info("Sleeping %ds until next cycle…", config.POLL_INTERVAL_SECONDS)
        time.sleep(config.POLL_INTERVAL_SECONDS)

    log.info("Bot stopped after %d cycle(s)", cycle)
