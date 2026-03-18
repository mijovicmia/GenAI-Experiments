"""Compute engineered features from a MarketSnapshot."""

from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean

from bot.data_models import EngineeredMarket, Market, MarketSnapshot, OrderBook, Trade
from bot.logger import get_logger

log = get_logger(__name__)

# Number of price levels to consider for liquidity depth
_DEPTH_LEVELS = 5
# Short / long window for momentum (in number of recent trades)
_MA_SHORT = 5
_MA_LONG = 20


def _implied_prob(ob: OrderBook) -> float:
    """Mid-price as implied probability (clamped to [0.01, 0.99])."""
    mid = ob.mid_price
    if mid is None:
        return 0.5
    return max(0.01, min(0.99, mid))


def _spread_bps(ob: OrderBook) -> float:
    """Best bid/ask spread in basis points."""
    if ob.best_bid is None or ob.best_ask is None or ob.best_bid <= 0:
        return 10_000.0
    mid = ob.mid_price or 1.0
    return ((ob.best_ask - ob.best_bid) / mid) * 10_000


def _momentum_score(trades: list[Trade]) -> float:
    """Simple momentum: MA_short − MA_long of recent trade prices.

    Returns a float in roughly [-1, 1] range when normalised by mid-price.
    Returns 0.0 if insufficient data.
    """
    prices = [t.price for t in sorted(trades, key=lambda t: t.timestamp)]
    if len(prices) < _MA_LONG:
        return 0.0
    ma_short = mean(prices[-_MA_SHORT:])
    ma_long = mean(prices[-_MA_LONG:])
    denominator = ma_long if ma_long else 1.0
    return (ma_short - ma_long) / denominator


def _liquidity_score(ob: OrderBook, depth: int = _DEPTH_LEVELS) -> float:
    """Normalised liquidity score in [0, 1] based on order book depth.

    Uses total notional (price × size) at the top N levels on each side.
    The raw value is log-normalised and clamped.
    """
    import math

    def notional(levels: list) -> float:
        return sum(lv.price * lv.size for lv in levels[:depth])

    bid_notional = notional(ob.bids)
    ask_notional = notional(ob.asks)
    total = bid_notional + ask_notional
    if total <= 0:
        return 0.0

    # log-scale: 10 USDC → ~0.1, 1000 USDC → ~0.75, 10000 USDC → ~1.0
    score = math.log10(total + 1) / 4.0
    return min(1.0, max(0.0, score))


def _time_to_expiry(market: Market) -> float:
    """Hours until market close. Returns 0 if already closed/unknown."""
    if market.close_time is None:
        return 0.0
    now = datetime.now(tz=timezone.utc)
    close = market.close_time
    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    delta = (close - now).total_seconds()
    return max(0.0, delta / 3600)


def build_features(snapshot: MarketSnapshot) -> list[EngineeredMarket]:
    """Return a list of EngineeredMarket, one per market in the snapshot."""
    results: list[EngineeredMarket] = []

    for market in snapshot.markets:
        ob = snapshot.orderbooks.get(market.id)
        trades = snapshot.trades.get(market.id, [])

        if ob is None:
            log.debug("Skipping %s — no orderbook data", market.id)
            continue

        yes_token_id = market.outcomes[0].token_id if market.outcomes else market.id

        try:
            em = EngineeredMarket(
                market_id=market.id,
                token_id=yes_token_id,
                name=market.question,
                implied_prob=_implied_prob(ob),
                spread_bps=_spread_bps(ob),
                momentum_score=_momentum_score(trades),
                liquidity_score=_liquidity_score(ob),
                time_to_expiry_hours=_time_to_expiry(market),
            )
            results.append(em)
        except Exception as exc:
            log.warning("Feature engineering failed for %s: %s", market.id, exc)

    log.debug("Feature engineering complete: %d markets processed", len(results))
    return results
