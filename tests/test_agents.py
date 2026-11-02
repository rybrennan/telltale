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
    assert out[0] == {
        "category": "billing",
        "priority": "high",
        "escalate": "no",
    }
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


def test_tasks_parse_and_probs():
    from telltale.agents.tasks import TASKS

    for t in TASKS.values():
        text = (
            "["
            + ",".join(
                '{"id":'
                + str(i)
                + ","
                + ",".join(f'"{f}":"{v[0]}"' for f, v in t.fields.items())
                + "}"
                for i in range(len(t.probes))
            )
            + "]"
        )
        a = t.parse(text)
        assert all(x is not None for x in a)
        p = t.probs(a)
        assert p.shape == (len(t.probes), t.n_classes)
        assert np.allclose(p.sum(axis=1), 1)


def test_eval_fault_only_hits_one_agent_after_onset():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "agent_eval",
        pathlib.Path(__file__).parent.parent / "scripts" / "agent_eval.py",
    )
    ae = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ae)
    for f in ae.FAULTS:
        base = [
            ae.setup("triage", None, -1, k, r)
            for r in range(ae.ROUNDS)
            for k in range(ae.N_AGENTS)
        ]
        got = [
            ae.setup("triage", f, 2, k, r)
            for r in range(ae.ROUNDS)
            for k in range(ae.N_AGENTS)
        ]
        changed = {
            (i // ae.N_AGENTS, i % ae.N_AGENTS)
            for i, (b, g) in enumerate(zip(base, got, strict=True))
            if b != g
        }
        assert changed == {(r, 2) for r in range(ae.ONSET, ae.ROUNDS)}
