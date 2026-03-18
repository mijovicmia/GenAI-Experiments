"""Assembles a full MarketSnapshot from the exchange client."""

from __future__ import annotations

from datetime import datetime

from bot.data_models import MarketSnapshot
from bot.exchange_client import ExchangeClient
from bot.logger import get_logger

log = get_logger(__name__)


def build_snapshot(client: ExchangeClient, max_markets: int = 50) -> MarketSnapshot:
    """Fetch live data and return a consistent snapshot.

    Args:
        client: Initialised ExchangeClient.
        max_markets: Upper bound on markets to process per cycle (performance guard).
    """
    log.info("Building market snapshot…")

    markets = client.list_markets(filters={"active_only": True})
    markets = [m for m in markets if not m.closed][:max_markets]

    orderbooks: dict = {}
    trades_map: dict = {}

    for market in markets:
        # Use the YES-token for price discovery (index 0 by convention)
        if not market.outcomes:
            continue
        yes_token = market.outcomes[0].token_id

        try:
            ob = client.get_market_orderbook(yes_token)
            orderbooks[market.id] = ob
        except Exception as exc:
            log.warning("Orderbook fetch failed for %s: %s", market.id, exc)

        try:
            trd = client.get_recent_trades(yes_token, limit=50)
            trades_map[market.id] = trd
        except Exception as exc:
            log.warning("Trades fetch failed for %s: %s", market.id, exc)

    positions = client.get_positions()
    account_state = client.get_account_state()

    snapshot = MarketSnapshot(
        markets=markets,
        orderbooks=orderbooks,
        trades=trades_map,
        positions=positions,
        account_state=account_state,
        timestamp=datetime.utcnow(),
    )

    log.info(
        "Snapshot built: %d markets, %d orderbooks, %d positions, cash=%.2f USDC",
        len(markets),
        len(orderbooks),
        len(positions),
        account_state.cash,
    )
    return snapshot
