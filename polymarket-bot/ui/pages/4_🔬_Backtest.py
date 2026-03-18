"""Backtest — visualise equity curve and performance metrics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Backtest | Polymarket Bot", page_icon="🔬", layout="wide")

RESULTS_DIR = Path(__file__).parent.parent.parent / "data" / "backtest_results"
EQUITY_CSV = RESULTS_DIR / "equity_curve.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"

st.title("🔬 Backtest Results")
st.caption(
    "Run `python scripts/run_backtest.py --data <snapshots.jsonl>` to generate results, "
    "or upload files directly below."
)

st.markdown("---")

# ── Load results ───────────────────────────────────────────────────────────────
tab_auto, tab_upload = st.tabs(["📂 Auto-load from disk", "⬆️ Upload results"])

equity_df: pd.DataFrame | None = None
metrics: dict | None = None

with tab_auto:
    if EQUITY_CSV.exists() and METRICS_JSON.exists():
        st.success(f"Results found at `{RESULTS_DIR}`", icon="✅")
        equity_df = pd.read_csv(EQUITY_CSV, parse_dates=["timestamp"])
        with open(METRICS_JSON) as f:
            metrics = json.load(f)
    else:
        st.info(
            "No results found yet. Run a backtest first:\n\n"
            "```bash\npython scripts/run_backtest.py --data data/snapshots.jsonl\n```",
            icon="ℹ️",
        )

with tab_upload:
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        up_equity = st.file_uploader("Equity curve CSV", type="csv", key="up_equity")
    with col_u2:
        up_metrics = st.file_uploader("Metrics JSON", type="json", key="up_metrics")

    if up_equity:
        equity_df = pd.read_csv(up_equity, parse_dates=["timestamp"])
    if up_metrics:
        metrics = json.load(up_metrics)

# ── Render results ─────────────────────────────────────────────────────────────
if metrics:
    st.markdown("---")
    st.subheader("Performance Summary")

    init = metrics.get("initial_cash", 10_000)
    final = metrics.get("final_equity", init)
    ret = metrics.get("total_return_pct", 0)
    dd = metrics.get("max_drawdown_pct", 0)
    pnl = metrics.get("realized_pnl", 0)
    cycles = metrics.get("cycles_run", 0)

    ret_delta = f"{ret:+.2f}%" if ret != 0 else None
    dd_color = "normal" if dd < 10 else "inverse"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Initial Cash", f"${init:,.2f}")
    c2.metric("Final Equity", f"${final:,.2f}", delta=ret_delta)
    c3.metric("Total Return", f"{ret:+.2f}%")
    c4.metric("Max Drawdown", f"{dd:.2f}%", delta=f"-{dd:.2f}%" if dd > 0 else None, delta_color="inverse")
    c5.metric("Cycles Run", cycles)

    st.markdown("---")

if equity_df is not None and not equity_df.empty:
    st.subheader("Equity Curve")

    # Main equity chart
    st.line_chart(
        equity_df.set_index("timestamp")["equity"],
        color="#4f8bf9",
        height=350,
    )

    # PnL breakdown
    pnl_tab, components_tab, raw_tab = st.tabs(["📈 PnL Breakdown", "🧩 Components", "🗃️ Raw Data"])

    with pnl_tab:
        if "realized_pnl" in equity_df.columns and "unrealized_pnl" in equity_df.columns:
            pnl_df = equity_df.set_index("timestamp")[["realized_pnl", "unrealized_pnl"]]
            st.area_chart(pnl_df, height=300, color=["#00c896", "#f9c74f"])
        else:
            st.info("PnL column not found in equity CSV.", icon="ℹ️")

    with components_tab:
        if "cash" in equity_df.columns:
            components = equity_df.set_index("timestamp")[
                [c for c in ["cash", "realized_pnl", "unrealized_pnl", "equity"] if c in equity_df.columns]
            ]
            st.line_chart(components, height=350)

    with raw_tab:
        st.dataframe(
            equity_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.DatetimeColumn("Time"),
                "equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
                "cash": st.column_config.NumberColumn("Cash", format="$%.2f"),
                "realized_pnl": st.column_config.NumberColumn("Realised PnL", format="$%.2f"),
                "unrealized_pnl": st.column_config.NumberColumn("Unrealised PnL", format="$%.2f"),
            },
        )

        col_dl, _ = st.columns([1, 3])
        with col_dl:
            st.download_button(
                "⬇️ Download CSV",
                data=equity_df.to_csv(index=False).encode(),
                file_name="equity_curve.csv",
                mime="text/csv",
            )

    # ── Drawdown chart ─────────────────────────────────────────────────────────
    if "equity" in equity_df.columns and len(equity_df) > 1:
        st.subheader("Drawdown")
        eq = equity_df["equity"]
        rolling_max = eq.cummax()
        drawdown = (eq - rolling_max) / rolling_max * 100
        dd_df = equity_df[["timestamp"]].copy()
        dd_df["drawdown_pct"] = drawdown.values
        st.area_chart(
            dd_df.set_index("timestamp")["drawdown_pct"],
            color="#f96b4f",
            height=200,
        )

elif equity_df is not None and equity_df.empty:
    st.warning("Equity curve CSV is empty.", icon="⚠️")
