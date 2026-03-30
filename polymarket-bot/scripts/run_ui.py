#!/usr/bin/env python3
"""Launch the Streamlit dashboard.

Usage:
    python scripts/run_ui.py
    # or directly:
    streamlit run ui/app.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(ROOT / "ui" / "app.py")],
    cwd=ROOT,
    check=True,
)
