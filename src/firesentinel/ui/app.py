"""Empty Streamlit shell for the FireSentinel interface."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render the intentionally empty Day 3 application shell."""
    st.set_page_config(page_title="FireSentinel", page_icon="🔥", layout="wide")
    st.title("FireSentinel")
    st.caption("UI shell ready — data, vision results, and agent decisions follow.")


if __name__ == "__main__":
    main()
