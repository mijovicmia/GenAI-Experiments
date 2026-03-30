"""Markets — live engineered feature data from Polymarket."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Markets | Polymarket Bot", page_icon="📊", layout="wide")

st.title("📊 Live Markets")
st.caption(
    "Fetches active markets from Polymarket, computes engineered features, "
    "and displays them here. Requires `CLAUDE_API_KEY` and network access."
)

# ── Controls ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    max_markets = st.slider("Max markets to fetch", min_value=5, max_value=100, value=20, step=5)
with col2:
    fetch_btn = st.button("🔄 Fetch Markets", type="primary", use_container_width=True)

st.markdown("---")


@st.cache_data(ttl=60, show_spinner="Fetching markets from Polymarket…")
def fetch_features(max_m: int) -> list[dict] | None:
    try:
        from bot.exchange_client import ExchangeClient
        from bot.data_fetcher import build_snapshot
        from bot.feature_engineering import build_features

        client = ExchangeClient()
        snapshot = build_snapshot(client, max_markets=max_m)
        features = build_features(snapshot)
        return [f.model_dump() for f in features]
    except Exception as exc:
        return {"error": str(exc)}


# ── Fetch ──────────────────────────────────────────────────────────────────────
if fetch_btn or "markets_data" in st.session_state:
    if fetch_btn:
        st.cache_data.clear()

    result = fetch_features(max_markets)

    if isinstance(result, dict) and "error" in result:
        st.error(f"Failed to fetch markets: {result['error']}", icon="🚨")
        st.info(
            "If running in stub mode (no Polymarket credentials), "
            "the exchange client returns empty data.",
            icon="ℹ️",
        )
    elif not result:
        st.warning("No markets returned. Check your Polymarket credentials.", icon="⚠️")
    else:
        df = pd.DataFrame(result)
        st.session_state["markets_data"] = df

        # ── Summary metrics ───────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Markets Fetched", len(df))
        c2.metric("Avg Implied Prob", f"{df['implied_prob'].mean():.3f}")
        c3.metric("Avg Liquidity Score", f"{df['liquidity_score'].mean():.3f}")
        c4.metric("Avg Spread (bps)", f"{df['spread_bps'].mean():.1f}")

        st.markdown("---")

        # ── Filters ───────────────────────────────────────────────────────────
        with st.expander("🔽 Filters", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            min_liq = fc1.slider("Min liquidity score", 0.0, 1.0, 0.0, 0.05)
            min_expiry = fc2.number_input("Min hours to expiry", min_value=0.0, value=0.0)
            sort_by = fc3.selectbox(
                "Sort by",
                ["liquidity_score", "implied_prob", "spread_bps", "momentum_score", "time_to_expiry_hours"],
            )

        filtered = df[
            (df["liquidity_score"] >= min_liq) &
            (df["time_to_expiry_hours"] >= min_expiry)
        ].sort_values(sort_by, ascending=False)

        st.caption(f"Showing {len(filtered)} of {len(df)} markets")

        # ── Charts ─────────────────────────────────────────────────────────────
        chart_tab, table_tab = st.tabs(["📈 Charts", "📋 Table"])

        with chart_tab:
            ch1, ch2 = st.columns(2)

            with ch1:
                st.subheader("Implied Probability Distribution")
                st.bar_chart(
                    filtered.set_index("name")["implied_prob"].head(20),
                    color="#4f8bf9",
                    height=300,
                )

            with ch2:
                st.subheader("Liquidity Score Distribution")
                st.bar_chart(
                    filtered.set_index("name")["liquidity_score"].head(20),
                    color="#00c896",
                    height=300,
                )

            st.subheader("Spread (bps) by Market")
            st.bar_chart(
                filtered.set_index("name")["spread_bps"].head(20),
                color="#f96b4f",
                height=250,
            )

            st.subheader("Momentum Score")
            momentum_df = filtered.set_index("name")["momentum_score"].head(20)
            st.bar_chart(momentum_df, color="#c8a000", height=250)

        with table_tab:
            st.dataframe(
                filtered[[
                    "name", "implied_prob", "spread_bps",
                    "momentum_score", "liquidity_score", "time_to_expiry_hours"
                ]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Market", width="large"),
                    "implied_prob": st.column_config.ProgressColumn(
                        "Implied Prob", min_value=0, max_value=1, format="%.3f"
                    ),
                    "liquidity_score": st.column_config.ProgressColumn(
                        "Liquidity", min_value=0, max_value=1, format="%.3f"
                    ),
                    "spread_bps": st.column_config.NumberColumn("Spread (bps)", format="%.1f"),
                    "momentum_score": st.column_config.NumberColumn("Momentum", format="%.4f"),
                    "time_to_expiry_hours": st.column_config.NumberColumn("Expiry (h)", format="%.1f"),
                },
            )
else:
    st.info("Click **Fetch Markets** to load live data from Polymarket.", icon="👆")

    # Show sample / demo data so the page isn't empty
    st.markdown("### Sample Feature Layout")
    sample = pd.DataFrame([
        {"name": "Will X win the election?", "implied_prob": 0.62, "spread_bps": 45.2,
         "momentum_score": 0.023, "liquidity_score": 0.78, "time_to_expiry_hours": 168.0},
        {"name": "Will BTC exceed $100k?", "implied_prob": 0.38, "spread_bps": 82.5,
         "momentum_score": -0.011, "liquidity_score": 0.55, "time_to_expiry_hours": 720.0},
        {"name": "Will the Fed cut rates?", "implied_prob": 0.71, "spread_bps": 30.1,
         "momentum_score": 0.041, "liquidity_score": 0.91, "time_to_expiry_hours": 48.0},
    ])
    st.dataframe(sample, use_container_width=True, hide_index=True)
