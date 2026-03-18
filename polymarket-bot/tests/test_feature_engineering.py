"""Unit tests for feature_engineering.py — all deterministic, no I/O."""

from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from bot.data_models import (
    AccountState,
    EngineeredMarket,
    Market,
    MarketSnapshot,
    OrderBook,
    Outcome,
    PriceLevel,
    Trade,
)
from bot.feature_engineering import (
    _implied_prob,
    _liquidity_score,
    _momentum_score,
    _spread_bps,
    _time_to_expiry,
    build_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ob(bids: list[tuple], asks: list[tuple], token_id: str = "tok1") -> OrderBook:
    return OrderBook(
        market_id="mkt1",
        token_id=token_id,
        bids=[PriceLevel(price=p, size=s) for p, s in bids],
        asks=[PriceLevel(price=p, size=s) for p, s in asks],
    )


def make_trades(prices: list[float]) -> list[Trade]:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        Trade(
            trade_id=str(i),
            market_id="mkt1",
            timestamp=base + timedelta(minutes=i),
            price=p,
            size=1.0,
            side="BUY",
        )
        for i, p in enumerate(prices)
    ]


# ---------------------------------------------------------------------------
# _implied_prob
# ---------------------------------------------------------------------------


def test_implied_prob_midpoint():
    ob = make_ob([(0.40, 100)], [(0.60, 100)])
    assert _implied_prob(ob) == pytest.approx(0.50)


def test_implied_prob_clamped_low():
    ob = make_ob([], [(0.001, 1)])
    # No bids → mid is None → returns 0.5
    assert _implied_prob(ob) == pytest.approx(0.5)


def test_implied_prob_one_sided_ask_only():
    ob = OrderBook(
        market_id="m", token_id="t",
        bids=[],
        asks=[PriceLevel(price=0.9, size=5)],
    )
    assert _implied_prob(ob) == pytest.approx(0.5)


def test_implied_prob_extremes_clamped():
    ob = make_ob([(0.001, 1)], [(0.002, 1)])
    prob = _implied_prob(ob)
    assert 0.01 <= prob <= 0.99


# ---------------------------------------------------------------------------
# _spread_bps
# ---------------------------------------------------------------------------


def test_spread_bps_basic():
    ob = make_ob([(0.40, 10)], [(0.42, 10)])
    # spread = 0.02, mid = 0.41 → spread_bps ≈ 48.8
    bps = _spread_bps(ob)
    assert bps == pytest.approx(0.02 / 0.41 * 10_000, rel=0.01)


def test_spread_bps_no_data():
    ob = make_ob([], [])
    assert _spread_bps(ob) == pytest.approx(10_000.0)


def test_spread_bps_tight():
    ob = make_ob([(0.499, 50)], [(0.501, 50)])
    bps = _spread_bps(ob)
    assert bps < 100  # tight market


# ---------------------------------------------------------------------------
# _momentum_score
# ---------------------------------------------------------------------------


def test_momentum_score_insufficient_data():
    trades = make_trades([0.5] * 10)
    assert _momentum_score(trades) == pytest.approx(0.0)


def test_momentum_score_uptrend():
    # Prices rising steadily — short MA > long MA → positive momentum
    prices = [0.3 + i * 0.01 for i in range(25)]
    trades = make_trades(prices)
    score = _momentum_score(trades)
    assert score > 0


def test_momentum_score_downtrend():
    # Prices falling steadily — short MA < long MA → negative momentum
    prices = [0.7 - i * 0.01 for i in range(25)]
    trades = make_trades(prices)
    score = _momentum_score(trades)
    assert score < 0


def test_momentum_score_flat():
    prices = [0.5] * 25
    trades = make_trades(prices)
    assert _momentum_score(trades) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _liquidity_score
# ---------------------------------------------------------------------------


def test_liquidity_score_empty():
    ob = make_ob([], [])
    assert _liquidity_score(ob) == pytest.approx(0.0)


def test_liquidity_score_range():
    ob = make_ob(
        [(0.45, 100), (0.44, 200), (0.43, 300)],
        [(0.55, 100), (0.56, 200), (0.57, 300)],
    )
    score = _liquidity_score(ob)
    assert 0.0 <= score <= 1.0


def test_liquidity_score_deep_market_higher():
    shallow_ob = make_ob([(0.5, 1)], [(0.51, 1)])
    deep_ob = make_ob(
        [(0.5, 1000)] * 5,
        [(0.51, 1000)] * 5,
    )
    assert _liquidity_score(deep_ob) > _liquidity_score(shallow_ob)


# ---------------------------------------------------------------------------
# _time_to_expiry
# ---------------------------------------------------------------------------


def test_time_to_expiry_future():
    market = Market(
        id="m1",
        question="Q?",
        close_time=datetime.now(tz=timezone.utc) + timedelta(hours=24),
    )
    hours = _time_to_expiry(market)
    assert 23.9 < hours < 24.1


def test_time_to_expiry_past():
    market = Market(
        id="m1",
        question="Q?",
        close_time=datetime.now(tz=timezone.utc) - timedelta(hours=1),
    )
    assert _time_to_expiry(market) == pytest.approx(0.0)


def test_time_to_expiry_none():
    market = Market(id="m1", question="Q?", close_time=None)
    assert _time_to_expiry(market) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_features integration
# ---------------------------------------------------------------------------


def test_build_features_returns_engineered_markets():
    market = Market(
        id="mkt1",
        question="Will X happen?",
        outcomes=[Outcome(name="Yes", token_id="tok1")],
        close_time=datetime.now(tz=timezone.utc) + timedelta(hours=48),
    )
    ob = make_ob([(0.4, 100)], [(0.6, 100)])
    trades = make_trades([0.5] * 25)
    snapshot = MarketSnapshot(
        markets=[market],
        orderbooks={"mkt1": ob},
        trades={"mkt1": trades},
        positions=[],
        account_state=AccountState(cash=1000.0),
    )

    result = build_features(snapshot)
    assert len(result) == 1
    em = result[0]
    assert isinstance(em, EngineeredMarket)
    assert em.market_id == "mkt1"
    assert 0 < em.implied_prob < 1
    assert em.spread_bps >= 0
    assert em.liquidity_score >= 0
    assert em.time_to_expiry_hours > 0


def test_build_features_skips_missing_orderbook():
    market = Market(
        id="mkt2",
        question="Another?",
        outcomes=[Outcome(name="Yes", token_id="tok2")],
    )
    snapshot = MarketSnapshot(
        markets=[market],
        orderbooks={},   # no orderbook
        trades={},
        positions=[],
        account_state=AccountState(cash=1000.0),
    )
    result = build_features(snapshot)
    assert result == []
