"""Claude API client — loads the skill and returns validated TradeActions."""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from bot import config
from bot.data_models import TradeAction
from bot.logger import get_logger

log = get_logger(__name__)

_SKILL_PATH = Path(__file__).parent / "skills" / "prediction_market_trader.skill.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8")


def _parse_actions(raw: str) -> list[TradeAction]:
    """Parse and validate the JSON array returned by Claude."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Claude returned invalid JSON: %s", exc)
        log.debug("Raw response: %s", raw[:500])
        return [TradeAction(action="HOLD", market_id=None, reason="Claude returned invalid JSON")]

    if not isinstance(data, list):
        log.error("Claude response is not a JSON array")
        return [TradeAction(action="HOLD", market_id=None, reason="Claude response was not a list")]

    actions: list[TradeAction] = []
    for item in data:
        try:
            action = TradeAction.model_validate(item)
            actions.append(action)
        except Exception as exc:
            log.warning("Skipping malformed action %s: %s", item, exc)

    if not actions:
        log.warning("No valid actions parsed from Claude response")
        return [TradeAction(action="HOLD", market_id=None, reason="No valid actions in response")]

    return actions


def get_trade_actions(skill_input: dict) -> list[TradeAction]:
    """Send market data to Claude and return a list of TradeActions.

    Args:
        skill_input: Dict matching the SkillInput schema.

    Returns:
        List of validated TradeAction objects.
    """
    if not config.CLAUDE_API_KEY:
        log.error("CLAUDE_API_KEY not set — returning HOLD")
        return [TradeAction(action="HOLD", market_id=None, reason="CLAUDE_API_KEY not configured")]

    skill_content = _load_skill()
    user_message = json.dumps(skill_input, indent=2, default=str)

    system_prompt = (
        f"{skill_content}\n\n"
        "Remember: respond with a JSON array only. No markdown, no explanation outside the JSON."
    )

    log.debug(
        "Sending %d markets and %d positions to Claude",
        len(skill_input.get("markets", [])),
        len(skill_input.get("positions", [])),
    )

    client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)

    try:
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as exc:
        log.error("Claude API error: %s", exc)
        return [TradeAction(action="HOLD", market_id=None, reason=f"Claude API error: {exc}")]

    raw = response.content[0].text.strip()

    # Strip markdown code fences if Claude wraps the JSON (defensive)
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    actions = _parse_actions(raw)
    log.info(
        "Claude returned %d action(s): %s",
        len(actions),
        [a.action for a in actions],
    )
    return actions
