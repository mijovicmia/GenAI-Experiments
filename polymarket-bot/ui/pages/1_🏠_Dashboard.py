"""Dashboard — account state, open positions, recent decisions."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

from ui.components.metrics_row import metrics_row

st.set_page_config(page_title="Dashboard | Polymarket Bot", page_icon="🏠", layout="wide")

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
TRADES_JSONL = LOGS_DIR / "paper_trades.jsonl"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 Dashboard")
st.caption(f"Last refreshed: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

if st.button("🔄 Refresh", type="secondary"):
    st.rerun()

st.markdown("---")


# ── Load trade log ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)
def load_trades() -> pd.DataFrame:
    if not TRADES_JSONL.exists():
        return pd.DataFrame()
    rows = []
    with open(TRADES_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp", ascending=False)


df = load_trades()

# ── Account metrics ────────────────────────────────────────────────────────────
st.subheader("Account State")

if df.empty:
    st.info("No paper trades logged yet. Start the bot with `python scripts/run_paper_trading.py`.", icon="ℹ️")
    total_trades = 0
    buys = sells = exits = holds = 0
    realised_pnl = 0.0
else:
    total_trades = len(df)
    buys = int((df["action"] == "BUY").sum())
    sells = int((df["action"] == "SELL").sum())
    exits = int((df["action"] == "EXIT").sum())
    holds = int((df["action"] == "HOLD").sum())

    # Approximate realised PnL from paper sizes
    buy_notional = df.loc[df["action"] == "BUY", "size"].fillna(0).astype(float).sum()
    sell_notional = df.loc[df["action"].isin(["SELL", "EXIT"]), "size"].fillna(0).astype(float).sum()
    realised_pnl = sell_notional - buy_notional

metrics_row([
    {"label": "Total Decisions", "value": total_trades},
    {"label": "BUY", "value": buys, "delta": None},
    {"label": "SELL / EXIT", "value": sells + exits},
    {"label": "HOLD", "value": holds},
    {"label": "Approx. Realised PnL", "value": f"${realised_pnl:+.2f}", "help": "sell notional − buy notional (paper)"},
])

st.markdown("---")

# ── Open paper positions ───────────────────────────────────────────────────────
st.subheader("Open Paper Positions")

if not df.empty:
    # Reconstruct approximate open positions from the trade log
    positions: dict[str, dict] = {}
    for _, row in df.sort_values("timestamp").iterrows():
        mid = row.get("market_id")
        if not mid or row["action"] == "HOLD":
            continue
        if row["action"] == "BUY":
            if mid not in positions:
                positions[mid] = {"side": row.get("side", "YES"), "size": 0.0, "entry_price": 0.0, "trades": 0}
            pos = positions[mid]
            new_size = pos["size"] + float(row.get("size") or 0)
            entry_price = row.get("limit_price") or 0.5
            if new_size > 0:
                pos["entry_price"] = (pos["size"] * pos["entry_price"] + float(row.get("size") or 0) * float(entry_price)) / new_size
            pos["size"] = new_size
            pos["trades"] += 1
        elif row["action"] in ("SELL", "EXIT"):
            if mid in positions:
                sell_size = float(row.get("size") or positions[mid]["size"])
                positions[mid]["size"] = max(0, positions[mid]["size"] - sell_size)
                if positions[mid]["size"] < 0.01:
                    del positions[mid]

    open_positions = [
        {
            "Market ID": mid,
            "Side": p["side"],
            "Size (USDC)": f"{p['size']:.2f}",
            "Avg Entry": f"{p['entry_price']:.4f}",
            "Trades": p["trades"],
        }
        for mid, p in positions.items()
        if p["size"] > 0.01
    ]

    if open_positions:
        st.dataframe(pd.DataFrame(open_positions), use_container_width=True, hide_index=True)
    else:
        st.success("No open positions.", icon="✅")
else:
    st.info("No data yet.", icon="ℹ️")

st.markdown("---")

# ── Recent decisions ────────────────────────────────────────────────────────────
st.subheader("Recent Decisions")

if not df.empty:
    recent = df.head(20).copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Colour-code actions
    def _action_badge(action: str) -> str:
        colours = {"BUY": "🟢", "SELL": "🟡", "EXIT": "🔴", "HOLD": "⚪"}
        return f"{colours.get(action, '')} {action}"

    recent["action"] = recent["action"].apply(_action_badge)
    display_cols = ["timestamp", "action", "market_id", "side", "size", "limit_price", "reason"]
    display_cols = [c for c in display_cols if c in recent.columns]
    st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)
else:
    st.info("No decisions logged yet.", icon="ℹ️")
