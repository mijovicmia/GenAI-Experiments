"""Unit tests for the exchange client (stub mode — no real API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from bot.data_models import AccountState, Order, OrderBook, Position


# ---------------------------------------------------------------------------
# ExchangeClient in stub mode (py-clob-client not installed / mocked)
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Return an ExchangeClient with _client set to None (stub mode)."""
    with patch("bot.exchange_client._CLOB_AVAILABLE", False):
        from bot.exchange_client import ExchangeClient
        c = ExchangeClient.__new__(ExchangeClient)
        c._client = None
        return c


def test_list_markets_stub_returns_empty(client):
    result = client.list_markets()
    assert result == []


def test_get_market_orderbook_stub_returns_empty(client):
    ob = client.get_market_orderbook("tok1")
    assert isinstance(ob, OrderBook)
    assert ob.bids == []
    assert ob.asks == []


def test_get_recent_trades_stub_returns_empty(client):
    trades = client.get_recent_trades("tok1")
    assert trades == []


def test_get_positions_stub_returns_empty(client):
    positions = client.get_positions()
    assert positions == []


def test_get_account_state_stub_returns_zero_cash(client):
    state = client.get_account_state()
    assert isinstance(state, AccountState)
    assert state.cash == 0.0


def test_place_order_stub_returns_stub_order(client):
    order = client.place_order(
        token_id="tok1",
        side="BUY",
        size=50.0,
        price=0.45,
        market_id="mkt1",
    )
    assert isinstance(order, Order)
    assert order.status == "stub"
    assert order.token_id == "tok1"
    assert order.side == "BUY"
    assert order.size == 50.0


def test_cancel_order_stub_does_not_raise(client):
    # Should log and return without raising
    client.cancel_order("order-abc")


# ---------------------------------------------------------------------------
# ExchangeClient with mocked py-clob-client
# ---------------------------------------------------------------------------


def test_list_markets_parses_response():
    """Verify list_markets correctly maps raw API response to Market objects."""
    with patch("bot.exchange_client._CLOB_AVAILABLE", True):
        from bot.exchange_client import ExchangeClient

        mock_clob = MagicMock()
        mock_clob.get_markets.return_value = {
            "data": [
                {
                    "condition_id": "mkt-abc",
                    "question": "Will it rain?",
                    "category": "Weather",
                    "end_date_iso": "2024-12-31T00:00:00",
                    "tokens": [
                        {"outcome": "Yes", "token_id": "tok-yes", "price": "0.6"},
                        {"outcome": "No", "token_id": "tok-no", "price": "0.4"},
                    ],
                    "minimum_tick_size": "0.01",
                    "active": True,
                    "closed": False,
                }
            ]
        }

        c = ExchangeClient.__new__(ExchangeClient)
        c._client = mock_clob

        markets = c.list_markets()
        assert len(markets) == 1
        m = markets[0]
        assert m.id == "mkt-abc"
        assert m.question == "Will it rain?"
        assert len(m.outcomes) == 2
        assert m.outcomes[0].name == "Yes"
        assert m.outcomes[0].token_id == "tok-yes"


def test_get_account_state_converts_wei():
    """Verify USDC balance is correctly converted from wei (÷ 1e6)."""
    with patch("bot.exchange_client._CLOB_AVAILABLE", True):
        from bot.exchange_client import ExchangeClient

        mock_clob = MagicMock()
        mock_clob.get_balance.return_value = "5000000000"  # 5000 USDC in wei
        mock_clob.get_positions.return_value = []

        c = ExchangeClient.__new__(ExchangeClient)
        c._client = mock_clob

        state = c.get_account_state()
        assert state.cash == pytest.approx(5000.0)
