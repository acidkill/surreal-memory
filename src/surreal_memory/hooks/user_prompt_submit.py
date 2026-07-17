"""UserPromptSubmit hook: inject learned reasoning strategies into the prompt.

SessionStart runs before any assistant turn exists, so the active model often
can't be resolved yet. From the second prompt on, the model is resolvable from
the transcript tail, so this hook injects model-appropriate reasoning strategies
that SessionStart may have missed. It shares the once-per-session marker with
SessionStart (whichever fires first wins), so the two never double-inject.

Opt-in via reasoning_training.injection_enabled.

Claude Code injects a UserPromptSubmit hook's context ONLY via the
``hookSpecificOutput.additionalContext`` JSON field on stdout (exit 0). Plain
stdout is echoed to the transcript but is NOT added to the model's context, so
the block is emitted inside that JSON envelope.

Usage as Claude Code hook:
    Reads JSON from stdin (session_id, transcript_path, cwd, prompt).
    Emits the reasoning block as hookSpecificOutput JSON on stdout (or nothing).
    Status messages go to stderr. Always exits 0 — never blocks the prompt.

Usage standalone:
    echo '{}' | python -m surreal_memory.hooks.user_prompt_submit
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def read_hook_input() -> dict[str, Any]:
    """Read Claude Code hook JSON from stdin (empty/malformed -> {})."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


def main() -> None:
    """Entry point for the UserPromptSubmit hook."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    hook_input = read_hook_input()

    from surreal_memory.engine.reasoning_injection import get_reasoning_context

    try:
        block = asyncio.run(get_reasoning_context(hook_input))
    except Exception:
        block = ""
        # Never block the prompt — degrade to no injection.
        print("[Surreal-Memory] UserPromptSubmit reasoning injection failed", file=sys.stderr)  # noqa: T201

    if block:
        # Context reaches the model ONLY through hookSpecificOutput.additionalContext
        # (plain stdout is transcript-only for this event).
        print(  # noqa: T201
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": block,
                    }
                }
            )
        )
    else:
        print("[Surreal-Memory] No reasoning strategies to inject", file=sys.stderr)  # noqa: T201

    sys.exit(0)


if __name__ == "__main__":
    main()
