"""Reusable metric row component."""

from __future__ import annotations
import streamlit as st


def metrics_row(items: list[dict]) -> None:
    """Render a row of st.metric cards.

    Args:
        items: list of dicts with keys: label, value, delta (optional), help (optional)
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.metric(
                label=item["label"],
                value=item["value"],
                delta=item.get("delta"),
                help=item.get("help"),
            )
