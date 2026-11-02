"""Probe batteries for agent monitoring, one per task type.

A task is a fixed set of probe inputs plus categorical answer fields. Every
agent labels every probe each round; `answers_to_probs` turns the labels into
a (probes, classes) probability array the detector already understands.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from telltale.agents import probes as _triage


@dataclass(frozen=True)
class Task:
    """A probe battery and the answer fields agents fill in for each probe."""

    name: str
    instruction: str
    fields: dict[str, tuple[str, ...]]
    probes: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        """Total labels across all fields."""
        return sum(len(v) for v in self.fields.values())

    def prompt(self) -> str:
        """The user prompt asking an agent to label every probe."""
        lines = [f"{i}. {t}" for i, t in enumerate(self.probes)]
        opts = "; ".join(
            f"{k}: one of {', '.join(v)}" for k, v in self.fields.items()
        )
        keys = ", ".join(f'"{k}": "..."' for k in self.fields)
        return (
            f"{self.instruction} For every item give {opts}.\n\n"
            + "\n".join(lines)
            + "\n\nReply with only a JSON list, one object per item: "
            + f'[{{"id": 0, {keys}}}, ...]'
        )

    def parse(self, text: str) -> list[dict | None]:
        """One dict per probe (None if missing); invalid labels become None."""
        out: list[dict | None] = [None] * len(self.probes)
        match = re.search(r"\[.*\]", text, flags=re.S)
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
            if 0 <= i < len(self.probes):
                out[i] = {
                    f: (
                        v
                        if (v := str(item.get(f, "")).strip().lower())
                        in allowed
                        else None
                    )
                    for f, allowed in self.fields.items()
                }
        return out

    def probs(self, answers: list[dict | None]) -> np.ndarray:
        """(probes, classes): one-hot per field, each field weighted 1/F."""
        p = np.zeros((len(answers), self.n_classes))
        w = 1.0 / len(self.fields)
        for i, ans in enumerate(answers):
            col = 0
            for f, allowed in self.fields.items():
                v = None if ans is None else ans.get(f)
                if v is None:
                    p[i, col : col + len(allowed)] = w / len(allowed)
                else:
                    p[i, col + allowed.index(v)] = w
                col += len(allowed)
        return p


TRIAGE = Task(
    name="triage",
    instruction="Triage each support ticket below.",
    fields=_triage.FIELDS,
    probes=_triage.PROBES,
)

EXPENSES = Task(
    name="expenses",
    instruction="Review each expense report line below against a typical "
    "company travel and expense policy.",
    fields={
        "decision": ("approve", "review", "reject"),
        "issue": (
            "none",
            "missing_receipt",
            "over_limit",
            "personal",
            "unclear",
            "other",
        ),
    },
    probes=(
        "Dinner with a client, $86, receipt attached.",
        "Hotel, 3 nights at $410 a night for a conference, receipt attached.",
        "Uber from airport to hotel, $54, no receipt.",
        "Team lunch for 8 people, $312, receipt attached.",
        "Gym membership, $65 per month.",
        "Economy flight to a sales meeting, $380, receipt attached.",
        "First-class upgrade on a domestic flight, $600.",
        "Coffee with a candidate during an interview, $12, no receipt.",
        "Laptop charger replacement, $79, receipt attached.",
        "Alcohol at a team offsite, $240, receipt attached.",
        "Parking at the office for a month, $180.",
        "Conference registration, $1,200, invoice attached.",
        "Movie tickets during a business trip, $28.",
        "Rental car for 2 days, $190, receipt attached.",
        "Home internet bill for remote work, $70.",
        "Gift for a client's wedding, $150.",
        "Meal on a travel day, $45, receipt is blurry.",
        "Software subscription for a design tool, $30 per month.",
        "Taxi home after working late, $38, receipt attached.",
        "Spa treatment at the conference hotel, $210.",
    ),
)

TASKS = {t.name: t for t in (TRIAGE, EXPENSES)}
