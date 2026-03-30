"""Config — display current risk parameters and environment settings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Config | Polymarket Bot", page_icon="⚙️", layout="wide")

st.title("⚙️ Configuration")
st.caption("Read-only view of the current bot configuration. Edit values in your `.env` file.")

st.markdown("---")

# ── Load config ────────────────────────────────────────────────────────────────
try:
    from bot import config

    cfg_ok = True
except Exception as exc:
    st.error(f"Failed to load bot config: {exc}", icon="🚨")
    cfg_ok = False

if cfg_ok:
    # ── Mode & timing ──────────────────────────────────────────────────────────
    st.subheader("Bot Mode")
    mode_col, interval_col = st.columns(2)
    with mode_col:
        mode_color = "🔴 LIVE" if config.MODE == "live" else "🟢 PAPER"
        st.metric("Mode", mode_color)
    with interval_col:
        st.metric("Poll Interval", f"{config.POLL_INTERVAL_SECONDS}s")

    st.markdown("---")

    # ── Risk parameters ────────────────────────────────────────────────────────
    st.subheader("Risk Parameters")

    risk_data = {
        "Parameter": [
            "Max Position per Market",
            "Max Daily Loss",
            "Max Open Trades",
            "Min Liquidity Score",
            "Market Cooldown After EXIT",
        ],
        "Value": [
            f"${config.MAX_POSITION_PER_MARKET:,.0f} USDC",
            f"${config.MAX_DAILY_LOSS:,.0f} USDC",
            str(config.MAX_OPEN_TRADES),
            f"{config.MIN_LIQUIDITY_SCORE:.2f}",
            f"{config.MARKET_COOLDOWN_HOURS:.1f} hours",
        ],
        "Env Var": [
            "MAX_POSITION_PER_MARKET",
            "MAX_DAILY_LOSS",
            "MAX_OPEN_TRADES",
            "MIN_LIQUIDITY_SCORE",
            "MARKET_COOLDOWN_HOURS",
        ],
        "Description": [
            "Maximum USDC notional per single market",
            "Bot pauses all new trades after this daily loss",
            "No new trades opened beyond this count",
            "Markets below this score are skipped",
            "Re-entry blocked for this window after exiting",
        ],
    }

    import pandas as pd
    st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Exchange settings ──────────────────────────────────────────────────────
    st.subheader("Exchange Settings")

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    ex_col1.metric("Base URL", config.POLYMARKET_BASE_URL)
    ex_col2.metric("Chain ID", config.POLYMARKET_CHAIN_ID)
    ex_col3.metric("Signature Type", config.POLYMARKET_SIGNATURE_TYPE)

    # Credential presence check (never show actual values)
    st.subheader("Credential Status")
    creds = {
        "CLAUDE_API_KEY": bool(config.CLAUDE_API_KEY),
        "POLYMARKET_PRIVATE_KEY": bool(config.POLYMARKET_PRIVATE_KEY),
        "POLYMARKET_FUNDER_ADDRESS": bool(config.POLYMARKET_FUNDER_ADDRESS),
        "POLYMARKET_API_KEY": bool(config.POLYMARKET_API_KEY),
        "POLYMARKET_API_SECRET": bool(config.POLYMARKET_API_SECRET),
        "POLYMARKET_PASSPHRASE": bool(config.POLYMARKET_PASSPHRASE),
    }

    cred_cols = st.columns(3)
    for i, (name, present) in enumerate(creds.items()):
        with cred_cols[i % 3]:
            status = "✅ Set" if present else "❌ Not set"
            required = name in ("CLAUDE_API_KEY",)
            st.metric(name, status, help="Required" if required else "Optional for paper mode")

    st.markdown("---")

    # ── Claude settings ────────────────────────────────────────────────────────
    st.subheader("Claude Settings")
    cl1, cl2 = st.columns(2)
    cl1.metric("Model", config.CLAUDE_MODEL)
    cl2.metric("Skill File", "bot/skills/prediction_market_trader.skill.md")

    st.markdown("---")

    # ── Logging ────────────────────────────────────────────────────────────────
    st.subheader("Logging")
    lg1, lg2 = st.columns(2)
    lg1.metric("Log Level", config.LOG_LEVEL)
    lg2.metric("Log File", config.LOG_FILE or "Console only")

    # ── Raw log tail ───────────────────────────────────────────────────────────
    log_path = Path(__file__).parent.parent.parent / "logs" / "bot.log"
    if log_path.exists():
        st.markdown("---")
        st.subheader("Recent Bot Log")
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-50:])
        st.code(tail, language=None)
    else:
        st.info("No bot.log file found yet. The log appears once the bot runs.", icon="ℹ️")
