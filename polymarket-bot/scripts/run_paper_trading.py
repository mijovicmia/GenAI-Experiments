#!/usr/bin/env python3
"""Launch the bot in paper-trading mode (no real orders sent)."""

import sys
from pathlib import Path

# Allow running from project root or scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.run_loop import run

if __name__ == "__main__":
    run(mode="paper")
