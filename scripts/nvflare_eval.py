#!/usr/bin/env python3
"""Score the NVFlare integration across heterogeneity and participation.

Measures detection delay and false-alarm rate. Runs examples/nvflare/job.py
in the NVFlare simulator over a grid and scores the aggregator's recorded
histories with the stage-4 protocol from the README:

    cells          alpha in {100, 0.5, 0.1} x participation in {1.0, 0.6}
    per cell       healthy train seeds 0, 1, 2, and a blur fault (1.5 px,
                   site-3, onset round 15, ten-round ramp) on seeds 0, 1, 2
    threshold      CUSUM healthy maximum after the reference epoch, seed 0,
                   per cell; it never sees a fault
    false alarms   counted on held-out healthy seeds 1 and 2, per client
                   contact
    delay          rounds from onset to the first flag on the faulted client

Every run records CUSUM with no threshold set; the threshold is applied
afterwards, so calibration and scoring use exactly the histories the
aggregator wrote.

    python scripts/nvflare_eval.py run --jobs 3   # every missing cell (~1 h)
    python scripts/nvflare_eval.py score          # tables + results json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
JOB = REPO / "examples" / "nvflare" / "job.py"
OUT = REPO / "results" / "nvflare"
SUMMARY = REPO / "results" / "nvflare_eval.json"

N_CLIENTS, ROUNDS, REFERENCE = 10, 30, (5, 15)
ALPHAS = (100.0, 0.5, 0.1)
PARTICIPATION = (1.0, 0.6)
SEEDS = (0, 1, 2)
FAULT = {"site": 3, "kind": "blur", "severity": 1.5, "onset": 15, "ramp": 10}


def run_name(alpha: float, part: float, faulted: bool, seed: int) -> str:
    """File stem of one run, e.g. `a0.5_p0.6_blur_s1`."""
    kind = FAULT["kind"] if faulted else "healthy"
    return f"a{alpha:g}_p{part:g}_{kind}_s{seed}"


def grid() -> list[tuple[float, float, bool, int]]:
    """Every (alpha, participation, faulted, seed) cell, in run order."""
    return [
        (a, p, f, s)
        for a in ALPHAS
        for p in PARTICIPATION
        for f in (False, True)
        for s in SEEDS
    ]


def run_one(alpha: float, part: float, faulted: bool, seed: int) -> str:
    """Run one simulation unless its history already exists.

    Args:
        alpha: Dirichlet label-skew concentration.
        part: Fraction of sites sampled per round.
        faulted: Whether site-3 carries the blur fault.
        seed: Training seed.

    Returns:
        A one-line status for the run log.
    """
    name = run_name(alpha, part, faulted, seed)
    history = OUT / f"{name}.json"
    if history.exists():
        return f"skip {name}"
    cmd = [
        sys.executable, str(JOB),
        "--n_clients", str(N_CLIENTS), "--num_rounds", str(ROUNDS),
        "--alpha", str(alpha), "--participation", str(part),
        "--train_seed", str(seed),
        "--reference_start", str(REFERENCE[0]),
        "--reference_stop", str(REFERENCE[1]),
        "--history_path", str(history), "--name", f"telltale-{name}",
    ]  # fmt: skip
    if faulted:
        cmd += [
            "--fault_site", str(FAULT["site"]),
            "--fault_kind", FAULT["kind"],
            "--fault_severity", str(FAULT["severity"]),
            "--fault_onset", str(FAULT["onset"]),
            "--fault_ramp", str(FAULT["ramp"]),
        ]  # fmt: skip
    log = OUT / f"{name}.log"
    with open(log, "w") as fh:
        rc = subprocess.run(
            cmd, cwd=JOB.parent, stdout=fh, stderr=subprocess.STDOUT
        ).returncode
    if rc != 0 or not history.exists():
        return f"FAIL {name} (rc {rc}; see {log})"
    return f"done {name}"


def load(name: str) -> dict:
    """Load a run's history with CUSUM as a (rounds, clients) float array."""
    h = json.loads((OUT / f"{name}.json").read_text())
    h["cusum"] = np.array(
        [[np.nan if v is None else v for v in row] for row in h["cusum"]],
        dtype=float,
    )
    return h


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes in n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score_cell(alpha: float, part: float) -> dict:
    """Score one cell of the grid.

    The threshold comes from healthy seed 0, false alarms from healthy seeds
    1 and 2, and detection delay from the three fault runs.

    Args:
        alpha: Dirichlet label-skew concentration.
        part: Fraction of sites sampled per round.

    Returns:
        The cell's threshold, false alarms, detections and delays.
    """
    stop = REFERENCE[1]
    cal = load(run_name(alpha, part, False, 0))["cusum"][stop:]
    thr = float(np.nanmax(cal))

    contacts = alarms = rounds_with_alarm = n_rounds = 0
    starved = []
    for s in SEEDS[1:]:
        c = load(run_name(alpha, part, False, s))["cusum"][stop:]
        contacts += int(np.isfinite(c).sum())
        alarms += int((c > thr).sum())
        rounds_with_alarm += int((c > thr).any(axis=1).sum())
        n_rounds += c.shape[0]
        starved.append(int(np.isnan(c).all(axis=0).sum()))

    row = FAULT["site"] - 1
    onset = FAULT["onset"]
    detections, delays, false_loc, contacts_to_flag = [], [], [], []
    for s in SEEDS:
        c = load(run_name(alpha, part, True, s))["cusum"]
        above = np.isfinite(c) & (c > np.nan_to_num(thr, nan=np.inf))
        hit = [r for r in range(onset, ROUNDS) if above[r, row]]
        detections.append(bool(hit))
        if hit:
            delays.append(hit[0] - onset)
            reported = np.isfinite(c[onset : hit[0] + 1, row])
            contacts_to_flag.append(int(reported.sum()))
        others = {
            k
            for r in range(onset, ROUNDS)
            for k in range(c.shape[1])
            if k != row and above[r, k]
        }
        false_loc.append(len(others))

    k = sum(detections)
    return {
        "alpha": alpha,
        "participation": part,
        "threshold": thr,
        "false_alarm_contacts": [alarms, contacts],
        "false_alarm_rate": alarms / contacts if contacts else None,
        "any_client_alarm_rounds": [rounds_with_alarm, n_rounds],
        "clients_without_reference_by_seed": starved,
        "detected": [k, len(SEEDS)],
        "detected_wilson95": wilson(k, len(SEEDS)),
        "delay_rounds": delays,
        "delay_contacts": contacts_to_flag,
        "false_localisations_by_seed": false_loc,
    }


def score() -> list[dict]:
    """Score every cell, write the summary json and print the table."""
    rows = [score_cell(a, p) for a in ALPHAS for p in PARTICIPATION]
    summary = {"protocol": __doc__, "fault": FAULT, "cells": rows}
    SUMMARY.write_text(json.dumps(summary, indent=2, default=float))
    print(
        "| alpha | participation | threshold | false alarms | blur detected"
        " | delay, rounds (contacts) | false localisations |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        fa, n = r["false_alarm_contacts"]
        lo, hi = r["detected_wilson95"]
        pairs = zip(r["delay_rounds"], r["delay_contacts"], strict=True)
        delays = ", ".join(f"{d} ({c})" for d, c in pairs) or "none"
        print(
            f"| {r['alpha']:g} | {r['participation']:g}"
            f" | {r['threshold']:.2f} | {fa}/{n} ({100 * fa / n:.1f}%)"
            f" | {r['detected'][0]}/{r['detected'][1]}"
            f" [{lo:.2f}, {hi:.2f}] | {delays}"
            f" | {r['false_localisations_by_seed']} |"
        )
    return rows


def main() -> int:
    """Run the grid or score it, per the command line."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("command", choices=["run", "score"])
    ap.add_argument(
        "--jobs", type=int, default=1, help="Simulations to run at once."
    )
    args = ap.parse_args()
    if args.command == "run":
        OUT.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for line in pool.map(lambda g: run_one(*g), grid()):
                print(line, flush=True)
        return 0
    score()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
