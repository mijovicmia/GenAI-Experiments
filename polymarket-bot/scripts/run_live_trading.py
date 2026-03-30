#!/usr/bin/env python3
"""Launch the bot in live-trading mode.

WARNING: This will send real orders to Polymarket. Ensure credentials and
risk limits are correctly configured in your .env file before running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bot import config

if config.MODE != "live":
    print(
        "Warning: MODE env var is not set to 'live'. "
        "Proceeding anyway with mode=live."
    )

from bot.run_loop import run

if __name__ == "__main__":
    run(mode="live")
