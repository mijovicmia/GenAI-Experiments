# Polymarket Claude Trading Bot

An AI-powered prediction market trading bot that uses **Claude** as its decision-making brain and **Polymarket** (CLOB) as the execution backend.

---

## Architecture Overview

```
MarketSnapshot
    │
    ▼
feature_engineering  →  EngineeredMarket[]
    │
    ▼
decision_engine  →  [calls Claude with skill]  →  TradeAction[]
    │
    ▼
risk_manager  →  enforced TradeAction[]
    │
    ▼
order_executor  →  paper log  OR  live CLOB order
```

### Key modules

| Module | Purpose |
|---|---|
| `bot/exchange_client.py` | Thin wrapper over `py-clob-client` |
| `bot/data_fetcher.py` | Assembles `MarketSnapshot` from live data |
| `bot/feature_engineering.py` | Computes implied prob, spread, momentum, liquidity |
| `bot/skills/prediction_market_trader.skill.md` | Claude skill definition |
| `bot/claude_client.py` | Sends data to Claude, parses `TradeAction[]` |
| `bot/decision_engine.py` | Builds `SkillInput`, calls Claude, filters responses |
| `bot/risk_manager.py` | Hard position/loss/cooldown limits |
| `bot/order_executor.py` | Paper-log or live order placement |
| `bot/backtester.py` | Historical simulation with the same pipeline |

---

## Prerequisites

- Python 3.11+
- A [Polymarket](https://polymarket.com) account with:
  - A private key (wallet)
  - Funder address (the wallet holding USDC on Polygon)
  - API credentials (derived from private key or pre-generated)
- An [Anthropic API key](https://console.anthropic.com)

---

## Installation

```bash
# Clone / navigate to project
cd polymarket-bot

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description |
|---|---|
| `POLYMARKET_PRIVATE_KEY` | Ethereum private key (hex, with 0x prefix) |
| `POLYMARKET_FUNDER_ADDRESS` | Wallet address holding USDC on Polygon |
| `POLYMARKET_API_KEY` | CLOB API key (optional — derived from private key if omitted) |
| `POLYMARKET_API_SECRET` | CLOB API secret |
| `POLYMARKET_PASSPHRASE` | CLOB API passphrase |
| `CLAUDE_API_KEY` | Anthropic API key |
| `MODE` | `paper` (default) or `live` |
| `POLL_INTERVAL_SECONDS` | Seconds between trading cycles (default: 60) |
| `MAX_POSITION_PER_MARKET` | Max USDC per market (default: 500) |
| `MAX_DAILY_LOSS` | Max USDC loss per day (default: 200) |
| `MAX_OPEN_TRADES` | Max simultaneous open positions (default: 10) |
| `MIN_LIQUIDITY_SCORE` | Skip markets below this liquidity score (default: 0.5) |

---

## Running in Paper Mode

Paper mode logs intended trades to `logs/paper_trades.csv` and `logs/paper_trades.jsonl` without sending any real orders.

```bash
# Only CLAUDE_API_KEY is required for paper mode
python scripts/run_paper_trading.py
```

---

## Running in Live Mode

**Warning:** Live mode sends real orders to Polymarket. Start with small position limits.

```bash
# Ensure MODE=live and all Polymarket credentials are set in .env
python scripts/run_live_trading.py
```

---

## Running a Backtest

Prepare a JSONL file with historical snapshots (one `MarketSnapshot` JSON per line), then:

```bash
python scripts/run_backtest.py \
    --data data/historical_snapshots.jsonl \
    --output data/backtest_results \
    --cash 10000
```

Output files:
- `data/backtest_results/equity_curve.csv` — equity at each time step
- `data/backtest_results/metrics.json` — summary statistics

### Snapshot format

Each line in the JSONL file must be a JSON object matching `MarketSnapshot`:

```json
{
  "timestamp": "2024-01-15T12:00:00",
  "markets": [...],
  "orderbooks": {"market_id": {"token_id": "...", "bids": [...], "asks": [...]}},
  "trades": {"market_id": [{"trade_id": "...", "timestamp": "...", "price": 0.5, "size": 10, "side": "BUY"}]},
  "positions": [],
  "account_state": {"cash": 10000.0}
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -f docker/Dockerfile -t polymarket-bot .

# Run in paper mode (pass env vars via --env-file)
docker run --env-file .env -v $(pwd)/logs:/app/logs polymarket-bot

# Run in live mode
docker run --env-file .env -e MODE=live -v $(pwd)/logs:/app/logs polymarket-bot
```

---

## Risk Disclaimers

- Prediction market trading involves real financial risk.
- This bot is experimental software — use at your own risk.
- Always start with paper mode and small position limits.
- Never store private keys in version control.
- The bot's AI decisions are not financial advice.

---

## Project Structure

```
polymarket-bot/
  bot/
    __init__.py
    config.py              # Env var loading + constraint accessors
    data_models.py         # Pydantic v2 models
    exchange_client.py     # Polymarket CLOB wrapper
    data_fetcher.py        # Build MarketSnapshot from live data
    feature_engineering.py # Compute implied_prob, spread, momentum, liquidity
    claude_client.py       # Call Claude API, parse TradeAction[]
    skills/
      prediction_market_trader.skill.md  # Claude skill definition
    decision_engine.py     # Orchestrate Claude call + filter results
    risk_manager.py        # Hard risk rules (position cap, loss cap, cooldown)
    order_executor.py      # Paper log or live order placement
    backtester.py          # Historical backtest engine
    logger.py              # Structured logging with secret scrubbing
    run_loop.py            # Main trading loop
  tests/
    test_exchange_client.py
    test_feature_engineering.py
    test_decision_engine.py
  scripts/
    run_paper_trading.py
    run_live_trading.py
    run_backtest.py
  docker/
    Dockerfile
    entrypoint.sh
  requirements.txt
  .env.example
  README.md
```
