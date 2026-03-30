"""Thin wrapper over the Polymarket CLOB API via py-clob-client."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from bot import config
from bot.data_models import (
    AccountState,
    Market,
    Order,
    OrderBook,
    Outcome,
    Position,
    PriceLevel,
    Trade,
)
from bot.logger import get_logger

log = get_logger(__name__)

# Lazy import so tests can mock without installing py-clob-client
try:
    from py_clob_client.client import ClobClient as _ClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        BookParams,
        LimitOrderArgs,
        OrderType,
    )
    from py_clob_client.order_builder.constants import BUY, SELL

    _CLOB_AVAILABLE = True
except ImportError:
    _CLOB_AVAILABLE = False
    log.warning(
        "py-clob-client not installed — ExchangeClient will run in stub mode."
    )


class ExchangeClientError(Exception):
    """Raised when the exchange returns an unexpected error."""


class ExchangeClient:
    """Wraps Polymarket CLOB providing typed methods used by the bot."""

    def __init__(self) -> None:
        if not _CLOB_AVAILABLE:
            self._client = None
            log.warning("ExchangeClient running in stub mode (no py-clob-client).")
            return

        self._client: _ClobClient = _ClobClient(
            config.POLYMARKET_BASE_URL,
            key=config.POLYMARKET_PRIVATE_KEY or None,
            chain_id=config.POLYMARKET_CHAIN_ID,
            signature_type=config.POLYMARKET_SIGNATURE_TYPE,
            funder=config.POLYMARKET_FUNDER_ADDRESS or None,
        )

        if config.POLYMARKET_API_KEY:
            creds = ApiCreds(
                api_key=config.POLYMARKET_API_KEY,
                api_secret=config.POLYMARKET_API_SECRET,
                api_passphrase=config.POLYMARKET_PASSPHRASE,
            )
            self._client.set_api_creds(creds)
        elif config.POLYMARKET_PRIVATE_KEY:
            # Derive L2 creds from private key
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)

        log.info("ExchangeClient initialised (base_url=%s)", config.POLYMARKET_BASE_URL)

    # ------------------------------------------------------------------
    # Public market data
    # ------------------------------------------------------------------

    def list_markets(self, filters: dict | None = None) -> list[Market]:
        """Return active markets from the CLOB."""
        if self._client is None:
            return []

        try:
            raw: dict = self._client.get_markets()
        except Exception as exc:
            raise ExchangeClientError(f"list_markets failed: {exc}") from exc

        results: list[Market] = []
        for item in raw.get("data", []):
            outcomes = [
                Outcome(
                    name=tok.get("outcome", ""),
                    token_id=tok.get("token_id", ""),
                    price=float(tok.get("price", 0)) if tok.get("price") else None,
                )
                for tok in item.get("tokens", [])
            ]
            market = Market(
                id=item.get("condition_id", item.get("id", "")),
                question=item.get("question", ""),
                category=item.get("category"),
                close_time=item.get("end_date_iso") or item.get("game_start_time"),
                outcomes=outcomes,
                tick_size=float(item.get("minimum_tick_size", 0.01)),
                active=item.get("active", True),
                closed=item.get("closed", False),
            )
            if filters:
                if filters.get("active_only") and not market.active:
                    continue
            results.append(market)

        log.debug("list_markets returned %d markets", len(results))
        return results

    def get_market_orderbook(self, token_id: str) -> OrderBook:
        """Fetch order book for a YES-token.

        Note: Polymarket order books are per token_id, not market_id.
        """
        if self._client is None:
            return OrderBook(market_id="", token_id=token_id)

        try:
            raw: Any = self._client.get_order_book(token_id)
        except Exception as exc:
            raise ExchangeClientError(
                f"get_market_orderbook({token_id}) failed: {exc}"
            ) from exc

        bids = [
            PriceLevel(price=float(b.price), size=float(b.size))
            for b in (raw.bids or [])
        ]
        asks = [
            PriceLevel(price=float(a.price), size=float(a.size))
            for a in (raw.asks or [])
        ]
        return OrderBook(
            market_id=getattr(raw, "market", token_id),
            token_id=token_id,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
        )

    def get_recent_trades(self, token_id: str, limit: int = 100) -> list[Trade]:
        """Return recent trades for a token (maps to CLOB trade history)."""
        if self._client is None:
            return []

        try:
            raw = self._client.get_last_trades_price(token_id)
        except Exception as exc:
            log.warning("get_recent_trades(%s) failed: %s", token_id, exc)
            return []

        trades: list[Trade] = []
        for item in (raw or [])[:limit]:
            trades.append(
                Trade(
                    trade_id=str(item.get("id", "")),
                    market_id=item.get("market", token_id),
                    timestamp=datetime.fromisoformat(
                        item.get("created_at", datetime.utcnow().isoformat())
                    ),
                    price=float(item.get("price", 0)),
                    size=float(item.get("size", 0)),
                    side=item.get("side", "BUY").upper(),
                )
            )
        return trades

    # ------------------------------------------------------------------
    # Account / position data (requires L2 auth)
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        """Return open positions from the CLOB."""
        if self._client is None:
            return []

        try:
            raw = self._client.get_positions()
        except Exception as exc:
            log.warning("get_positions failed: %s", exc)
            return []

        positions: list[Position] = []
        for item in raw or []:
            side_raw = item.get("side", "YES").upper()
            side = "YES" if side_raw in ("YES", "BUY") else "NO"
            positions.append(
                Position(
                    market_id=item.get("market", ""),
                    token_id=item.get("asset_id", ""),
                    side=side,
                    size=float(item.get("size", 0)),
                    entry_price=float(item.get("avg_price", 0)),
                    current_price=float(item.get("cur_price", 0)),
                )
            )
        return positions

    def get_account_state(self) -> AccountState:
        """Return cash balance and PnL summary."""
        if self._client is None:
            return AccountState(cash=0.0)

        try:
            balance_wei = self._client.get_balance()
            cash = int(balance_wei) / 1e6  # convert from USDC wei
        except Exception as exc:
            log.warning("get_account_state: balance fetch failed: %s", exc)
            cash = 0.0

        positions = self.get_positions()
        unrealized = sum(p.unrealized_pnl for p in positions)
        return AccountState(cash=cash, unrealized_pnl=unrealized)

    # ------------------------------------------------------------------
    # Order management (requires L2 auth)
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side: str,       # "BUY" | "SELL"
        size: float,
        price: float,
        market_id: str = "",
    ) -> Order:
        """Place a limit order on the CLOB."""
        if self._client is None or not _CLOB_AVAILABLE:
            log.info(
                "[STUB] place_order token=%s side=%s size=%.4f price=%.4f",
                token_id, side, size, price,
            )
            return Order(
                order_id=f"stub-{int(time.time())}",
                market_id=market_id,
                token_id=token_id,
                side=side,
                size=size,
                price=price,
                status="stub",
            )

        clob_side = BUY if side.upper() == "BUY" else SELL

        try:
            order_args = LimitOrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=clob_side,
            )
            signed = self._client.create_order(order_args)
            resp = self._client.post_order(signed, OrderType.GTC)
        except Exception as exc:
            raise ExchangeClientError(f"place_order failed: {exc}") from exc

        order_id = resp.get("orderID", resp.get("id", ""))
        log.info(
            "Order placed: id=%s token=%s side=%s size=%.4f price=%.4f",
            order_id, token_id, side, size, price,
        )
        return Order(
            order_id=order_id,
            market_id=market_id,
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            status=resp.get("status", "open"),
        )

    def cancel_order(self, order_id: str) -> None:
        """Cancel an open order."""
        if self._client is None:
            log.info("[STUB] cancel_order id=%s", order_id)
            return

        try:
            self._client.cancel(order_id)
            log.info("Order cancelled: id=%s", order_id)
        except Exception as exc:
            raise ExchangeClientError(f"cancel_order({order_id}) failed: {exc}") from exc
