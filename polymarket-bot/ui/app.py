"""Polymarket Claude Trading Bot — Streamlit Dashboard.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Polymarket Claude Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://polymarket.com/og-image.png", use_container_width=True)
    st.title("Polymarket Claude Bot")
    st.caption("AI-powered prediction market trading")

    st.markdown("---")
    st.markdown(
        """
        **Pages**
        - 🏠 Dashboard — account & positions
        - 📋 Trade Log — paper trade history
        - 📊 Markets — live feature data
        - 🔬 Backtest — historical simulation
        - ⚙️ Config — risk parameters
        """
    )
    st.markdown("---")
    st.caption("Navigate using the sidebar pages →")

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("📈 Polymarket Claude Trading Bot")
st.markdown(
    """
    Welcome to the **Polymarket Claude Bot** dashboard. Use the sidebar to navigate between views.

    | Page | Description |
    |---|---|
    | 🏠 **Dashboard** | Account state, open positions, and recent decisions |
    | 📋 **Trade Log** | Full history of paper trades with filters |
    | 📊 **Markets** | Live engineered market features from Polymarket |
    | 🔬 **Backtest** | Visualise equity curve and metrics from historical runs |
    | ⚙️ **Config** | View and understand current risk configuration |
    """
)

st.info(
    "**Getting started:** Copy `.env.example` → `.env`, fill in your `CLAUDE_API_KEY`, "
    "then run `python scripts/run_paper_trading.py` to start generating trade data.",
    icon="💡",
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mode", "Paper", help="Set MODE=live in .env for live trading")
with col2:
    log_path = Path(__file__).parent.parent / "logs" / "paper_trades.jsonl"
    trade_count = sum(1 for _ in open(log_path)) if log_path.exists() else 0
    st.metric("Paper Trades Logged", trade_count)
with col3:
    backtest_path = Path(__file__).parent.parent / "data" / "backtest_results" / "metrics.json"
    st.metric("Backtest Results", "✅ Available" if backtest_path.exists() else "None yet")
