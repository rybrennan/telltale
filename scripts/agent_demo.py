#!/usr/bin/env python3
"""First test of telltale on LLM agents: does it name the agent that changed?

Five triage agents (Claude Haiku via `claude -p`, each with slightly
different instructions) answer the same 20 probe tickets every round. In fault
runs one agent changes from round ONSET: either its instructions are quietly
edited (a policy change) or its model is swapped (a model update). Healthy
runs have no change. The detector is telltale's own: per-agent divergence
from the leave-one-out peer consensus, accumulated by a CUSUM against the
agent's own reference rounds.

    python scripts/agent_demo.py run      # calls agents; resumes from cache
    python scripts/agent_demo.py score    # thresholds, false alarms, detections

Every raw reply is cached under results/agents/, so scoring never calls a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from telltale.agents.backends import claude_cli  # noqa: E402
from telltale.agents.probes import (  # noqa: E402
    PROBES,
    answers_to_probs,
    parse_answers,
    prompt_for,
)
from telltale.detect import cusum, level_series  # noqa: E402

OUT = REPO / "results" / "agents"
ROUNDS, ONSET, REFERENCE = 10, 5, slice(1, 5)
BASE = (
    "You are a customer support triage agent for a company that sells "
    "software subscriptions and ships physical products."
)
AGENTS = (
    BASE,
    BASE + " You work the overnight queue.",
    BASE + " You handle tickets from business customers.",
    BASE + " You are careful and thorough.",
    BASE + " You value resolving tickets quickly.",
)
FAULT_AGENT = 2
POLICY_EDIT = (
    " Updated guidance: treat refund, charge and billing questions as account "
    "issues, and escalate only tickets that mention legal action."
)
MODEL_DEFAULT, MODEL_SWAP = "haiku", "sonnet"
RUNS = {
    **{f"healthy_{s}": None for s in range(4)},
    **{f"policy_{s}": "policy" for s in range(2)},
    **{f"model_{s}": "model" for s in range(2)},
}


def agent_setup(run_fault: str | None, k: int, r: int) -> tuple[str, str]:
    """System prompt and model for agent k in round r of a run."""
    system, model = AGENTS[k], MODEL_DEFAULT
    if run_fault and k == FAULT_AGENT and r >= ONSET:
        if run_fault == "policy":
            system = system + POLICY_EDIT
        else:
            model = MODEL_SWAP
    return system, model


def reply_path(run: str, r: int, k: int) -> Path:
    """Cache file of one raw reply."""
    return OUT / run / f"r{r:02d}_a{k}.txt"


def call_one(run: str, fault: str | None, r: int, k: int) -> str:
    """Get one reply, from cache if present."""
    p = reply_path(run, r, k)
    if p.exists() and p.read_text().strip():
        return "cached"
    system, model = agent_setup(fault, k, r)
    for attempt in range(3):
        try:
            text = claude_cli(system, prompt_for(), model=model)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            return "ok"
        except Exception as e:  # retry transient CLI failures
            err = str(e)
    return f"FAIL {run} r{r} a{k}: {err[:120]}"


def run_all(jobs: int) -> None:
    """Call every agent for every round of every run (resumable)."""
    tasks = [
        (run, fault, r, k)
        for run, fault in RUNS.items()
        for r in range(ROUNDS)
        for k in range(len(AGENTS))
    ]
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for i, status in enumerate(pool.map(lambda t: call_one(*t), tasks)):
            if status.startswith("FAIL") or i % 25 == 0:
                print(f"{i + 1}/{len(tasks)} {status}", flush=True)


def probs_for(run: str) -> np.ndarray:
    """(rounds, agents, probes, classes) array for one run."""
    rows = []
    for r in range(ROUNDS):
        rows.append([
            answers_to_probs(parse_answers(reply_path(run, r, k).read_text()))
            for k in range(len(AGENTS))
        ])
    return np.array(rows)


def cusum_for(run: str) -> np.ndarray:
    """(rounds, agents) CUSUM of each agent's level against its own reference."""
    return cusum(level_series(probs_for(run))["level"], REFERENCE)


def score() -> dict:
    """Thresholds from healthy runs, held-out false alarms, detections."""
    healthy = [r for r, f in RUNS.items() if f is None]
    S = {run: cusum_for(run) for run in RUNS}
    stop = REFERENCE.stop
    # False alarms: threshold from the other healthy runs, scored on one held out.
    far = []
    for h in healthy:
        thr = max(np.nanmax(S[o][stop:]) for o in healthy if o != h)
        s = S[h][stop:]
        far.append({
            "held_out": h, "threshold": float(thr),
            "agent_rounds_flagged": int((s > thr).sum()),
            "agent_rounds": int(np.isfinite(s).sum()),
            "run_alarmed": bool((s > thr).any()),
        })
    thr_all = max(np.nanmax(S[h][stop:]) for h in healthy)
    det = []
    for run, fault in RUNS.items():
        if fault is None:
            continue
        s = S[run]
        hits = [r for r in range(ONSET, ROUNDS) if s[r, FAULT_AGENT] > thr_all]
        others = sorted({
            k for r in range(stop, ROUNDS) for k in range(len(AGENTS))
            if k != FAULT_AGENT and s[r, k] > thr_all
        })
        det.append({
            "run": run, "fault": fault, "caught": bool(hits),
            "rounds_to_flag": hits[0] - ONSET if hits else None,
            "healthy_agents_flagged": others,
            "faulted_agent_peak": float(np.nanmax(s[ONSET:, FAULT_AGENT])),
        })
    parse_fail = sum(
        a is None
        for run in RUNS for r in range(ROUNDS) for k in range(len(AGENTS))
        for a in parse_answers(reply_path(run, r, k).read_text())
    )
    out = {
        "setup": {"agents": len(AGENTS), "probes": len(PROBES),
                  "rounds": ROUNDS, "onset": ONSET,
                  "reference_rounds": [REFERENCE.start, REFERENCE.stop - 1],
                  "model": MODEL_DEFAULT, "fault_agent": FAULT_AGENT},
        "threshold_all_healthy": float(thr_all),
        "false_alarms": far, "detections": det,
        "unparsed_answers": parse_fail,
    }
    (REPO / "results" / "agent_demo.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    if args.command == "run":
        run_all(args.jobs)
    else:
        score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
