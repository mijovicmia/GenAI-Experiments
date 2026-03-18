"""Trade Log — full filterable paper trade history."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Trade Log | Polymarket Bot", page_icon="📋", layout="wide")

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
TRADES_JSONL = LOGS_DIR / "paper_trades.jsonl"
TRADES_CSV = LOGS_DIR / "paper_trades.csv"


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
    for col in ["size", "limit_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📋 Trade Log")

df = load_trades()

if df.empty:
    st.warning("No paper trades logged yet. Start the bot to generate data.", icon="⚠️")
    st.stop()

# ── Summary bar ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", len(df))
col2.metric("Unique Markets", df["market_id"].nunique() if "market_id" in df.columns else 0)
col3.metric("Date Range",
    f"{df['timestamp'].min().strftime('%b %d')} – {df['timestamp'].max().strftime('%b %d')}"
    if len(df) > 0 else "—"
)
col4.metric("Actions", ", ".join(df["action"].unique()) if "action" in df.columns else "—")

st.markdown("---")

# ── Filters ─────────────────────────────────────────────────────────────────────
with st.expander("🔽 Filters", expanded=True):
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        all_actions = sorted(df["action"].dropna().unique().tolist()) if "action" in df.columns else []
        selected_actions = st.multiselect("Action", all_actions, default=all_actions)

    with fcol2:
        all_markets = sorted(df["market_id"].dropna().unique().tolist()) if "market_id" in df.columns else []
        selected_markets = st.multiselect("Market ID", all_markets, placeholder="All markets")

    with fcol3:
        date_range = st.date_input(
            "Date range",
            value=(df["timestamp"].min().date(), df["timestamp"].max().date()),
            key="date_range",
        )

# Apply filters
filtered = df.copy()
if selected_actions:
    filtered = filtered[filtered["action"].isin(selected_actions)]
if selected_markets:
    filtered = filtered[filtered["market_id"].isin(selected_markets)]
if len(date_range) == 2:
    start, end = date_range
    filtered = filtered[
        (filtered["timestamp"].dt.date >= start) & (filtered["timestamp"].dt.date <= end)
    ]

st.caption(f"Showing {len(filtered)} of {len(df)} records")

# ── Action breakdown chart ──────────────────────────────────────────────────────
st.subheader("Action Breakdown")

if not filtered.empty and "action" in filtered.columns:
    action_counts = filtered["action"].value_counts().reset_index()
    action_counts.columns = ["Action", "Count"]

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.bar_chart(action_counts.set_index("Action"), color="#4f8bf9")
    with table_col:
        st.dataframe(action_counts, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Trade size distribution ─────────────────────────────────────────────────────
if "size" in filtered.columns and filtered["size"].notna().any():
    st.subheader("Trade Size Distribution (USDC)")
    active_trades = filtered[filtered["action"].isin(["BUY", "SELL", "EXIT"])]["size"].dropna()
    if not active_trades.empty:
        st.area_chart(active_trades.sort_index(), color="#00c896")

st.markdown("---")

# ── Full table ─────────────────────────────────────────────────────────────────
st.subheader("All Records")

display = filtered.copy()
display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Colour action column with icons
action_icons = {"BUY": "🟢 BUY", "SELL": "🟡 SELL", "EXIT": "🔴 EXIT", "HOLD": "⚪ HOLD"}
if "action" in display.columns:
    display["action"] = display["action"].map(lambda x: action_icons.get(x, x))

col_order = ["timestamp", "action", "market_id", "side", "size", "limit_price", "reason"]
col_order = [c for c in col_order if c in display.columns]

st.dataframe(
    display[col_order],
    use_container_width=True,
    hide_index=True,
    column_config={
        "size": st.column_config.NumberColumn("Size (USDC)", format="$%.2f"),
        "limit_price": st.column_config.NumberColumn("Limit Price", format="%.4f"),
    },
)

# ── Download ────────────────────────────────────────────────────────────────────
st.markdown("---")
if TRADES_CSV.exists():
    with open(TRADES_CSV, "rb") as f:
        st.download_button(
            label="⬇️ Download CSV",
            data=f,
            file_name="paper_trades.csv",
            mime="text/csv",
        )
