#!/usr/bin/env python3
"""How much the results depend on which healthy seed sets the threshold.

The protocol calibrates each statistic on one healthy seed (the lowest) and
counts false alarms on the others. This script repeats that with every healthy
seed in turn as the calibration seed, the rest held out, and reports the
threshold, the held-out false-alarm rate and detection on the pooled
mid-severity population (one fault of each kind at its middle severity, on
every faulted client), per heterogeneity level and statistic. Scoring only; no
runs.

    python scripts/rotate_calibration.py
    python scripts/rotate_calibration.py --results /path/to/results
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

MID = {"label_noise": 0.3, "blur": 1.0, "gain": 0.5, "bias": 0.5}
STATS = (
    "level_cusum",
    "level_z",
    "update_norm_cusum",
    "update_loo_cusum",
    "fedrr",
    "fedrr_steps",
)


def rotate(results: Path) -> list[dict]:
    """Score every calibration-seed choice; return one row per choice."""
    sf.RESULTS = results
    families: dict = {}
    for f in sorted(results.glob("probes_*.json")):
        d, probs, upd = sf.load(f.stem)
        if (
            d["config"]["dataset"] != "fashion_mnist"
            or d.get("contact")
            or not sf.in_main_grid(d)
        ):
            continue
        families.setdefault(sf.family(d), {})[d["config"]["train_seed"]] = (
            d,
            sf.scored(d, probs, upd),
        )
    faults: dict = {}
    for f in sorted(results.glob("fault_*.json")):
        d, probs, upd = sf.load(f.stem)
        spec = d["fault"]
        if (
            d["config"]["dataset"] != "fashion_mnist"
            or d.get("contact")
            or not sf.in_main_grid(d)
            or spec["ramp"] != 10
            or MID.get(spec["kind"]) != spec["severity"]
        ):
            continue
        faults.setdefault(sf.family(d), []).append(
            (d, sf.scored(d, probs, upd))
        )
    rows = []
    for fam, runs in families.items():
        alpha = next(iter(runs.values()))[0]["config"]["dirichlet_alpha"]
        seeds = sorted(runs)
        for name in STATS:
            fam_faults = [
                (d, st) for d, st in faults.get(fam, []) if name in st
            ]
            if any(name not in runs[s][1] for s in seeds) or not fam_faults:
                continue
            for c in seeds:
                cd, cst = runs[c]
                classes = sf.platform_classes(cd)
                thr = sf.calibrate_by_class(
                    cst[name], classes, name, sf.BURN_IN
                )
                far = np.mean(
                    [
                        sf.node_round_rate(
                            sf.flags_by_class(runs[s][1][name], thr, classes),
                            runs[s][1][name],
                            sf.BURN_IN,
                        )
                        for s in seeds
                        if s != c
                    ]
                )
                outs = [
                    sf.score_flags(
                        sf.flags_by_class(st[name], thr, classes),
                        sf.BURN_IN,
                        d["fault"]["node"],
                        d["fault"]["onset"],
                    )
                    for d, st in fam_faults
                ]
                rows.append(
                    {
                        "alpha": alpha,
                        "stat": name,
                        "calibration_seed": c,
                        "threshold": float(thr["ship"].value),
                        "node_round_far": float(far),
                        "detected": int(sum(o.detected for o in outs)),
                        "fault_runs": len(outs),
                        "detected_client2": int(
                            sum(
                                o.detected
                                for o, (d, _) in zip(
                                    outs, fam_faults, strict=True
                                )
                                if d["fault"]["node"] == 2
                            )
                        ),
                        "fault_runs_client2": sum(
                            d["fault"]["node"] == 2 for d, _ in fam_faults
                        ),
                    }
                )
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rows = rotate(args.results)
    print(
        "| alpha | stat | calib seed | threshold | held-out FAR % | detected |"
    )
    print("|---|---|---|---|---|---|")
    for r in sorted(
        rows, key=lambda r: (r["alpha"], r["stat"], r["calibration_seed"])
    ):
        print(
            f"| {r['alpha']} | {r['stat']} | {r['calibration_seed']} | "
            f"{r['threshold']:.3f} | {100 * r['node_round_far']:.1f} | "
            f"{r['detected']}/{r['fault_runs']} "
            f"(client 2: {r['detected_client2']}/{r['fault_runs_client2']}) |"
        )
    out = args.out or args.results / "calibration_rotation.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
