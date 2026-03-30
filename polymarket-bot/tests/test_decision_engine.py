"""Unit tests for Claude response parsing and decision_engine filtering."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from bot.claude_client import _parse_actions
from bot.data_models import (
    AccountState,
    EngineeredMarket,
    MarketSnapshot,
    Outcome,
    Market,
    TradeAction,
)
from bot.decision_engine import decide


# ---------------------------------------------------------------------------
# _parse_actions — Claude response parsing
# ---------------------------------------------------------------------------


def test_parse_actions_valid():
    raw = json.dumps([
        {"action": "BUY", "market_id": "mkt1", "side": "YES", "size": 50.0,
         "limit_price": 0.45, "reason": "Good edge"},
        {"action": "HOLD", "market_id": None, "reason": "No edge elsewhere"},
    ])
    actions = _parse_actions(raw)
    assert len(actions) == 2
    assert actions[0].action == "BUY"
    assert actions[0].market_id == "mkt1"
    assert actions[1].action == "HOLD"


def test_parse_actions_invalid_json():
    actions = _parse_actions("not json {{")
    assert len(actions) == 1
    assert actions[0].action == "HOLD"
    assert "invalid JSON" in actions[0].reason.lower() or "JSON" in actions[0].reason


def test_parse_actions_not_a_list():
    actions = _parse_actions(json.dumps({"action": "BUY", "reason": "x"}))
    assert len(actions) == 1
    assert actions[0].action == "HOLD"


def test_parse_actions_skips_malformed_items():
    raw = json.dumps([
        {"action": "INVALID_ACTION", "reason": "bad"},  # invalid enum
        {"action": "HOLD", "market_id": None, "reason": "ok"},
    ])
    actions = _parse_actions(raw)
    # Malformed item is skipped, HOLD is retained
    assert any(a.action == "HOLD" for a in actions)


def test_parse_actions_empty_list():
    actions = _parse_actions("[]")
    assert len(actions) == 1
    assert actions[0].action == "HOLD"


# ---------------------------------------------------------------------------
# decision_engine.decide — filtering
# ---------------------------------------------------------------------------


def _make_snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        markets=[
            Market(id="mkt1", question="Q1?", outcomes=[Outcome(name="Yes", token_id="tok1")]),
            Market(id="mkt2", question="Q2?", outcomes=[Outcome(name="Yes", token_id="tok2")]),
        ],
        orderbooks={},
        trades={},
        positions=[],
        account_state=AccountState(cash=5000.0),
    )


def _make_engineered(market_ids: list[str]) -> list[EngineeredMarket]:
    return [
        EngineeredMarket(
            market_id=mid,
            token_id=f"tok_{mid}",
            name=f"Market {mid}",
            implied_prob=0.45,
            spread_bps=50.0,
            momentum_score=0.1,
            liquidity_score=0.8,
            time_to_expiry_hours=48.0,
        )
        for mid in market_ids
    ]


def test_decide_filters_unknown_market():
    snapshot = _make_snapshot()
    engineered = _make_engineered(["mkt1", "mkt2"])
    constraints = {"max_position_per_market": 500, "max_daily_loss": 200, "max_open_trades": 10}

    claude_actions = [
        TradeAction(action="BUY", market_id="unknown_market", side="YES", size=50, limit_price=0.45, reason="test"),
        TradeAction(action="HOLD", market_id=None, reason="nothing else"),
    ]

    with patch("bot.decision_engine.claude_client.get_trade_actions", return_value=claude_actions):
        result = decide(snapshot, engineered, constraints)

    market_ids_in_result = [a.market_id for a in result if a.market_id]
    assert "unknown_market" not in market_ids_in_result


def test_decide_filters_negative_size():
    snapshot = _make_snapshot()
    engineered = _make_engineered(["mkt1"])
    constraints = {"max_position_per_market": 500, "max_daily_loss": 200, "max_open_trades": 10}

    # Use model_construct to bypass Pydantic validation and simulate raw Claude output
    claude_actions = [
        TradeAction.model_construct(action="BUY", market_id="mkt1", side="YES", size=-10, limit_price=0.45, reason="test"),
    ]

    with patch("bot.decision_engine.claude_client.get_trade_actions", return_value=claude_actions):
        result = decide(snapshot, engineered, constraints)

    # Only HOLD should remain (fallback)
    assert all(a.action == "HOLD" for a in result)


def test_decide_filters_bad_limit_price():
    snapshot = _make_snapshot()
    engineered = _make_engineered(["mkt1"])
    constraints = {"max_position_per_market": 500, "max_daily_loss": 200, "max_open_trades": 10}

    # Use model_construct to bypass Pydantic validation and simulate raw Claude output
    claude_actions = [
        TradeAction.model_construct(action="BUY", market_id="mkt1", side="YES", size=50, limit_price=1.5, reason="test"),
    ]

    with patch("bot.decision_engine.claude_client.get_trade_actions", return_value=claude_actions):
        result = decide(snapshot, engineered, constraints)

    assert all(a.action == "HOLD" for a in result)


def test_decide_passes_valid_action():
    snapshot = _make_snapshot()
    engineered = _make_engineered(["mkt1", "mkt2"])
    constraints = {"max_position_per_market": 500, "max_daily_loss": 200, "max_open_trades": 10}

    claude_actions = [
        TradeAction(action="BUY", market_id="mkt1", side="YES", size=50, limit_price=0.45, reason="good edge"),
    ]

    with patch("bot.decision_engine.claude_client.get_trade_actions", return_value=claude_actions):
        result = decide(snapshot, engineered, constraints)

    buys = [a for a in result if a.action == "BUY"]
    assert len(buys) == 1
    assert buys[0].market_id == "mkt1"


def test_decide_returns_hold_when_all_filtered():
    snapshot = _make_snapshot()
    engineered = _make_engineered(["mkt1"])
    constraints = {"max_position_per_market": 500, "max_daily_loss": 200, "max_open_trades": 10}

    # All Claude actions are for unknown markets
    claude_actions = [
        TradeAction(action="BUY", market_id="ghost", side="YES", size=50, limit_price=0.5, reason="bad"),
    ]

    with patch("bot.decision_engine.claude_client.get_trade_actions", return_value=claude_actions):
        result = decide(snapshot, engineered, constraints)

    assert len(result) >= 1
    assert result[-1].action == "HOLD"
