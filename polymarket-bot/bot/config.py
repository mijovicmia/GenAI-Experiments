"""Configuration loader — reads from environment variables or config.yaml."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=False)


# ---------------------------------------------------------------------------
# Exchange (Polymarket)
# ---------------------------------------------------------------------------
POLYMARKET_PRIVATE_KEY: str = os.environ.get("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS: str = os.environ.get("POLYMARKET_FUNDER_ADDRESS", "")
POLYMARKET_API_KEY: str = os.environ.get("POLYMARKET_API_KEY", "")
POLYMARKET_API_SECRET: str = os.environ.get("POLYMARKET_API_SECRET", "")
POLYMARKET_PASSPHRASE: str = os.environ.get("POLYMARKET_PASSPHRASE", "")
POLYMARKET_BASE_URL: str = os.environ.get(
    "POLYMARKET_BASE_URL", "https://clob.polymarket.com"
)
POLYMARKET_CHAIN_ID: int = int(os.environ.get("POLYMARKET_CHAIN_ID", "137"))
POLYMARKET_SIGNATURE_TYPE: int = int(
    os.environ.get("POLYMARKET_SIGNATURE_TYPE", "1")
)

# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------
CLAUDE_API_KEY: str = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

# ---------------------------------------------------------------------------
# Bot behaviour
# ---------------------------------------------------------------------------
MODE: str = os.environ.get("MODE", "paper")  # "paper" | "live"
POLL_INTERVAL_SECONDS: int = int(os.environ.get("POLL_INTERVAL_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Risk constraints
# ---------------------------------------------------------------------------
MAX_POSITION_PER_MARKET: float = float(
    os.environ.get("MAX_POSITION_PER_MARKET", "500")
)
MAX_DAILY_LOSS: float = float(os.environ.get("MAX_DAILY_LOSS", "200"))
MAX_OPEN_TRADES: int = int(os.environ.get("MAX_OPEN_TRADES", "10"))
MIN_LIQUIDITY_SCORE: float = float(os.environ.get("MIN_LIQUIDITY_SCORE", "0.5"))
MARKET_COOLDOWN_HOURS: float = float(
    os.environ.get("MARKET_COOLDOWN_HOURS", "2")
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE: str | None = os.environ.get("LOG_FILE", "logs/bot.log") or None


def load_constraints() -> dict:
    """Return risk constraints as a plain dict (used by decision engine & risk manager)."""
    return {
        "max_position_per_market": MAX_POSITION_PER_MARKET,
        "max_daily_loss": MAX_DAILY_LOSS,
        "max_open_trades": MAX_OPEN_TRADES,
        "min_liquidity_score": MIN_LIQUIDITY_SCORE,
        "market_cooldown_hours": MARKET_COOLDOWN_HOURS,
    }


def validate() -> None:
    """Raise if any required credential is missing for the selected mode."""
    if MODE == "live":
        missing = [
            name
            for name, val in [
                ("POLYMARKET_PRIVATE_KEY", POLYMARKET_PRIVATE_KEY),
                ("POLYMARKET_FUNDER_ADDRESS", POLYMARKET_FUNDER_ADDRESS),
                ("CLAUDE_API_KEY", CLAUDE_API_KEY),
            ]
            if not val
        ]
        if missing:
            raise EnvironmentError(
                f"Missing required env vars for live mode: {missing}"
            )
    if not CLAUDE_API_KEY:
        raise EnvironmentError("CLAUDE_API_KEY is required in all modes")
