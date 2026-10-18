#!/usr/bin/env python3
"""Run-level false alarms, pooled calibration, per-client false flags.

Three questions the node-round false-alarm rate does not answer:

1. How often does a whole healthy run raise at least one alarm? With the
   threshold set to one healthy run's maximum, a held-out run of the same
   kind exceeds it about half the time by construction (two exchangeable
   runs, either can hold the larger maximum).
2. What changes if the threshold is the maximum over five healthy runs and
   the sixth is held out? A held-out run then exceeds it with probability
   about 1/6 under exchangeability.
3. When a fault run is detected, how many healthy clients were also flagged,
   and was the faulted client flagged before any healthy one (the strict
   rule)?

Both protocols are scored on the main grid (Fashion-MNIST, every client
reports every round, ramp 10). Fault runs are scored at every severity and
separately on the pooled mid-severity population used by
rotate_calibration.py. Under pooled-five calibration a fault run is scored
only with the threshold whose five calibration seeds exclude its own seed.
Scoring only; no runs.

    python scripts/review4_scores.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import score_faults as sf  # noqa: E402
from rotate_calibration import MID  # noqa: E402

from telltale.metrics import Threshold  # noqa: E402

STATS = (
    "level_cusum",
    "level_z",
    "update_norm_cusum",
    "update_loo_cusum",
    "fedrr",
    "fedrr_steps",
)


def pooled_threshold(series: list[np.ndarray], classes, name) -> dict:
    """Per-class maximum over several healthy runs after burn-in."""
    per_run = [
        sf.calibrate_by_class(s, classes, name, sf.BURN_IN) for s in series
    ]
    return {
        c: Threshold(
            name,
            max(t[c].value for t in per_run),
            sf.BURN_IN,
            f"max over {len(series)} runs",
            sum(t[c].n_calibration for t in per_run),
        )
        for c in per_run[0]
    }


def run_alarm(fl: np.ndarray) -> bool:
    """Any flag after burn-in."""
    return bool(fl[sf.BURN_IN :].any())


def fault_outcome(fl: np.ndarray, target: int, onset: int) -> dict:
    """Loose and strict detection, and healthy clients flagged."""
    out = sf.score_flags(fl, sf.BURN_IN, target, onset)
    others = np.ones(fl.shape[1], dtype=bool)
    others[target] = False
    strict = False
    if out.detected:
        # Strict: no healthy client flagged at or before the detection round.
        strict = not fl[sf.BURN_IN : out.detection_round + 1, others].any()
    return {
        "detected": out.detected,
        "strict": strict,
        "ttd": out.time_to_detection,
        "false_node_rounds": out.false_localisations,
        "false_clients": int(fl[sf.BURN_IN :, others].any(axis=0).sum()),
    }


def summarise(outs: list[dict]) -> dict:
    """Totals over a list of fault outcomes."""
    hits = [o for o in outs if o["detected"]]
    return {
        "n": len(outs),
        "detected": len(hits),
        "strict": sum(o["strict"] for o in outs),
        "ttd_median": float(np.median([o["ttd"] for o in hits]))
        if hits
        else None,
        "false_clients_mean": float(np.mean([o["false_clients"] for o in outs]))
        if outs
        else None,
        "false_node_rounds_mean": float(
            np.mean([o["false_node_rounds"] for o in outs])
        )
        if outs
        else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--onset",
        type=int,
        default=15,
        help="score only fault runs with this onset (the onset cells share "
        "their healthy runs' family with the main grid)",
    )
    args = p.parse_args()
    sf.RESULTS = args.results

    families: dict = {}
    for f in sorted(args.results.glob("probes_*.json")):
        d, probs, upd = sf.load(f.stem)
        c = d["config"]
        if c["dataset"] != "fashion_mnist" or d.get("contact"):
            continue
        families.setdefault(sf.family(d), {})[c["train_seed"]] = (
            d,
            sf.scored(d, probs, upd),
        )
    faults: dict = {}
    for f in sorted(args.results.glob("fault_*.json")):
        d, probs, upd = sf.load(f.stem)
        if (
            d["config"]["dataset"] != "fashion_mnist"
            or d.get("contact")
            or d["fault"]["ramp"] != 10
            or d["fault"]["onset"] != args.onset
        ):
            continue
        faults.setdefault(sf.family(d), []).append(
            (d, sf.scored(d, probs, upd))
        )

    rows = []
    for fam, runs in families.items():
        seeds = sorted(runs)
        if len(seeds) < 6 or fam not in faults:
            continue
        d0 = runs[seeds[0]][0]
        alpha = d0["config"]["dirichlet_alpha"]
        classes = sf.platform_classes(d0)
        for name in STATS:
            if any(name not in runs[s][1] for s in seeds):
                continue
            fam_faults = [(d, st) for d, st in faults[fam] if name in st]
            for protocol in ("single", "pooled5"):
                healthy_far, healthy_run = [], []
                outs_all, outs_mid = [], []
                for s in seeds:
                    if protocol == "single":
                        # s calibrates; the other five are held out.
                        thr = sf.calibrate_by_class(
                            runs[s][1][name], classes, name, sf.BURN_IN
                        )
                        held = [h for h in seeds if h != s]
                        scored_faults = fam_faults
                    else:
                        # s is held out; the other five calibrate.
                        thr = pooled_threshold(
                            [runs[h][1][name] for h in seeds if h != s],
                            classes,
                            name,
                        )
                        held = [s]
                        scored_faults = [
                            (d, st)
                            for d, st in fam_faults
                            if d["config"]["train_seed"] == s
                        ]
                    for h in held:
                        series = runs[h][1][name]
                        fl = sf.flags_by_class(series, thr, classes)
                        healthy_far.append(
                            sf.node_round_rate(fl, series, sf.BURN_IN)
                        )
                        healthy_run.append(run_alarm(fl))
                    for d, st in scored_faults:
                        spec = d["fault"]
                        o = fault_outcome(
                            sf.flags_by_class(st[name], thr, classes),
                            spec["node"],
                            spec["onset"],
                        )
                        o["node"] = spec["node"]
                        outs_all.append(o)
                        if MID.get(spec["kind"]) == spec["severity"]:
                            outs_mid.append(o)
                # "single" repeats every fault run once per calibration seed;
                # report per-choice averages so counts stay on one run each.
                k = len(seeds) if protocol == "single" else 1
                rows.append(
                    {
                        "alpha": alpha,
                        "partition_seed": d0["config"]["partition_seed"],
                        "clip_norm": d0["config"].get("clip_norm"),
                        "onset": args.onset,
                        "stat": name,
                        "protocol": protocol,
                        "healthy_runs_scored": len(healthy_run),
                        "node_round_far": float(np.mean(healthy_far)),
                        "run_alarm_rate": float(np.mean(healthy_run)),
                        "run_alarms": int(sum(healthy_run)),
                        "expected_run_alarm_rate": 0.5
                        if protocol == "single"
                        else 1 / 6,
                        "per_choice": k,
                        "all": summarise(outs_all),
                        "mid": summarise(outs_mid),
                        "mid_client2": summarise(
                            [o for o in outs_mid if o["node"] == 2]
                        ),
                    }
                )

    print(
        "| alpha | partition | clip | stat | protocol | node-round FAR | "
        "healthy runs alarmed | "
        "mid detected | mid strict | healthy clients flagged per fault run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(
        rows,
        key=lambda r: (
            r["alpha"],
            r["partition_seed"],
            str(r["clip_norm"]),
            r["stat"],
            r["protocol"],
        ),
    ):
        m, k = r["mid"], r["per_choice"]
        print(
            f"| {r['alpha']} | {r['partition_seed']} | {r['clip_norm']} | "
            f"{r['stat']} | {r['protocol']} | "
            f"{100 * r['node_round_far']:.1f}% | "
            f"{r['run_alarms']}/{r['healthy_runs_scored']} | "
            f"{m['detected'] / k:.1f}/{m['n'] // k} | "
            f"{m['strict'] / k:.1f}/{m['n'] // k} | "
            f"{m['false_clients_mean']:.2f} |"
        )
    out = args.out or args.results / (
        "review4_scores.json"
        if args.onset == 15
        else f"review4_scores_onset{args.onset}.json"
    )
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
