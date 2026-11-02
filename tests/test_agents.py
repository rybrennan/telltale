"""Tests for the agent probe battery and its fit to the existing detector."""

import numpy as np

from telltale.agents.probes import (
    FIELDS,
    N_CLASSES,
    PROBES,
    answers_to_probs,
    parse_answers,
    prompt_for,
)
from telltale.detect import cusum, level_series


def _reply(labels):
    items = [
        {"id": i, "category": c, "priority": p, "escalate": e}
        for i, (c, p, e) in enumerate(labels)
    ]
    import json

    return "Here you go:\n```json\n" + json.dumps(items) + "\n```"


def test_prompt_lists_every_probe_and_label():
    text = prompt_for()
    for i, probe in enumerate(PROBES):
        assert f"{i}. {probe}" in text
    for labels in FIELDS.values():
        for label in labels:
            assert label in text


def test_parse_handles_fences_and_bad_labels():
    labels = [("billing", "high", "no")] * len(PROBES)
    labels[1] = ("nonsense", "low", "yes")
    out = parse_answers(_reply(labels))
    assert out[0] == {"category": "billing", "priority": "high", "escalate": "no"}
    assert out[1]["category"] is None and out[1]["priority"] == "low"
    assert parse_answers("no json here") == [None] * len(PROBES)


def test_probs_rows_sum_to_one_and_missing_is_uniform():
    out = parse_answers(_reply([("billing", "high", "no")] * len(PROBES)))
    out[3] = None
    p = answers_to_probs(out)
    assert p.shape == (len(PROBES), N_CLASSES)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[3][:5], (1 / 3) / 5)


def test_detector_flags_the_agent_that_changes():
    rng = np.random.default_rng(0)
    rounds, agents = 10, 5
    cats = FIELDS["category"]
    base = [rng.choice(cats) for _ in PROBES]
    probs = []
    for r in range(rounds):
        row = []
        for k in range(agents):
            labels = []
            for i, c in enumerate(base):
                flip = rng.random() < 0.05
                if k == 2 and r >= 5 and i < 10:
                    c = "other"
                elif flip:
                    c = rng.choice(cats)
                labels.append((c, "medium", "no"))
            row.append(answers_to_probs(parse_answers(_reply(labels))))
        probs.append(row)
    s = cusum(level_series(np.array(probs))["level"], slice(1, 5))
    peak = np.nanmax(s[5:], axis=0)
    assert int(np.argmax(peak)) == 2
