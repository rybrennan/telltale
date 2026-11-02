"""Run an agent: the Claude Code CLI in headless mode (`claude -p`).

Uses the operator's Claude plan through the installed CLI; no API key.
"""

from __future__ import annotations

import subprocess
import tempfile


def claude_cli(system: str, prompt: str, model: str = "haiku",
               timeout: int = 180) -> str:
    """Return one agent reply from `claude -p` with a fixed system prompt."""
    cmd = [
        "claude", "-p", "--model", model, "--system-prompt", system,
        "--no-session-persistence", "--tools", "", "--setting-sources", "",
        prompt,
    ]
    # Empty stdin (the CLI appends stdin to the prompt) and a neutral working
    # directory, so no project CLAUDE.md, settings or hooks reach the agent.
    res = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, cwd=tempfile.gettempdir(),
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip()[:500])
    return res.stdout
