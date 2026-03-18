"""Pydantic v2 data models for markets, orders, positions, and account state."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Exchange primitives
# ---------------------------------------------------------------------------


class PriceLevel(BaseModel):
    price: float
    size: float


class OrderBook(BaseModel):
    market_id: str
    token_id: str
    bids: list[PriceLevel] = Field(default_factory=list)
    asks: list[PriceLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class Trade(BaseModel):
    trade_id: str
    market_id: str
    timestamp: datetime
    price: float
    size: float
    side: Literal["BUY", "SELL"]


class Outcome(BaseModel):
    name: str
    token_id: str
    price: float | None = None


class Market(BaseModel):
    id: str  # condition_id / market_id
    question: str
    category: str | None = None
    close_time: datetime | None = None
    outcomes: list[Outcome] = Field(default_factory=list)
    tick_size: float | None = None
    active: bool = True
    closed: bool = False

    @field_validator("close_time", mode="before")
    @classmethod
    def parse_close_time(cls, v: object) -> object:
        if isinstance(v, (int, float)):
            return datetime.utcfromtimestamp(v)
        return v


class Position(BaseModel):
    market_id: str
    token_id: str
    side: Literal["YES", "NO"]
    size: float
    entry_price: float
    current_price: float

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.size


class AccountState(BaseModel):
    cash: float  # USDC available
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class Order(BaseModel):
    order_id: str
    market_id: str
    token_id: str
    side: Literal["BUY", "SELL"]
    size: float
    price: float
    status: str = "open"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Aggregated snapshot (output of data_fetcher)
# ---------------------------------------------------------------------------


class MarketSnapshot(BaseModel):
    markets: list[Market]
    orderbooks: dict[str, OrderBook]  # keyed by market_id
    trades: dict[str, list[Trade]]    # keyed by market_id
    positions: list[Position]
    account_state: AccountState
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Feature-engineered representation
# ---------------------------------------------------------------------------


class EngineeredMarket(BaseModel):
    market_id: str
    token_id: str
    name: str
    implied_prob: float = Field(ge=0.0, le=1.0)
    spread_bps: float = Field(ge=0.0)
    momentum_score: float
    liquidity_score: float = Field(ge=0.0, le=1.0)
    time_to_expiry_hours: float


# ---------------------------------------------------------------------------
# Claude skill I/O
# ---------------------------------------------------------------------------


class SkillInputMarket(BaseModel):
    market_id: str
    name: str
    implied_prob: float
    spread_bps: float
    momentum_score: float
    liquidity_score: float
    time_to_expiry_hours: float


class SkillInputPosition(BaseModel):
    market_id: str
    side: Literal["YES", "NO"]
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float


class SkillInput(BaseModel):
    timestamp: str
    markets: list[SkillInputMarket]
    positions: list[SkillInputPosition]
    constraints: dict


class TradeAction(BaseModel):
    action: Literal["BUY", "SELL", "EXIT", "HOLD"]
    market_id: str | None = None
    side: Literal["YES", "NO"] | None = None
    size: float | None = Field(default=None, ge=0)
    limit_price: float | None = Field(default=None, ge=0, le=1)
    reason: str
