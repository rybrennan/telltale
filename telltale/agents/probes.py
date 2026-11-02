"""A fixed probe battery for support-ticket triage agents.

Each probe is a short customer ticket. An agent labels every ticket with a
category, a priority and whether to escalate. The labels are categorical, so
each answer becomes a one-hot row and the detector compares answer
distributions exactly as it compares class probabilities in federated
learning.
"""

from __future__ import annotations

import json
import re

import numpy as np

FIELDS: dict[str, tuple[str, ...]] = {
    "category": ("billing", "technical", "account", "shipping", "other"),
    "priority": ("low", "medium", "high"),
    "escalate": ("yes", "no"),
}

PROBES: tuple[str, ...] = (
    "I was charged twice for my subscription this month.",
    "The app crashes every time I open the settings page.",
    "How do I change the email address on my account?",
    "My package says delivered but it isn't here.",
    "Your product ruined my event and I'm talking to my lawyer.",
    "Can I get a refund for the annual plan I bought yesterday?",
    "I can't log in; the password reset email never arrives.",
    "The order arrived damaged and I need a replacement before Friday.",
    "Do you offer a discount for nonprofits?",
    "Someone made purchases on my account that I didn't make.",
    "The export to CSV feature has been broken for a week.",
    "Please delete my account and all my data.",
    "My invoice shows the wrong company name.",
    "Tracking hasn't updated in ten days.",
    "The new update deleted all my saved projects.",
    "I'd like to upgrade to the business plan.",
    "Your website is down and my whole team can't work.",
    "Can I change the shipping address on an order I placed an hour ago?",
    "I want to cancel my subscription; it's too expensive.",
    "The mobile app shows a different balance than the website.",
)

N_CLASSES = sum(len(v) for v in FIELDS.values())


def prompt_for(probes: tuple[str, ...] = PROBES) -> str:
    """The user prompt that asks an agent to label every probe ticket."""
    lines = [f"{i}. {t}" for i, t in enumerate(probes)]
    opts = "; ".join(f"{k}: one of {', '.join(v)}" for k, v in FIELDS.items())
    return (
        "Triage each support ticket below. For every ticket give "
        f"{opts}.\n\n" + "\n".join(lines) + "\n\n"
        'Reply with only a JSON list, one object per ticket: '
        '[{"id": 0, "category": "...", "priority": "...", "escalate": "..."}, ...]'
    )


def parse_answers(text: str, n_probes: int = len(PROBES)) -> list[dict | None]:
    """Parse an agent's JSON reply into one dict per probe (None if missing).

    Tolerates prose or code fences around the JSON list and drops values that
    are not among the allowed labels.
    """
    match = re.search(r"\[.*\]", text, flags=re.S)
    out: list[dict | None] = [None] * n_probes
    if not match:
        return out
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            i = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if not 0 <= i < n_probes:
            continue
        clean = {}
        for field, allowed in FIELDS.items():
            v = str(item.get(field, "")).strip().lower()
            clean[field] = v if v in allowed else None
        out[i] = clean
    return out


def answers_to_probs(answers: list[dict | None]) -> np.ndarray:
    """(probes, classes) array: one-hot per field, each field weighted 1/F.

    A missing or invalid field spreads its weight uniformly over that field's
    labels, so a bad reply looks maximally uncertain rather than confident.
    """
    probs = np.zeros((len(answers), N_CLASSES))
    w = 1.0 / len(FIELDS)
    for i, ans in enumerate(answers):
        col = 0
        for field, allowed in FIELDS.items():
            v = None if ans is None else ans.get(field)
            if v is None:
                probs[i, col : col + len(allowed)] = w / len(allowed)
            else:
                probs[i, col + allowed.index(v)] = w
            col += len(allowed)
    return probs
