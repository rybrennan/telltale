#!/usr/bin/env python3
"""FedRR's CUSUM allowance: the printed formula against the one used here.

The FedRR paper prints the update as S + Z - d/2, but its own control limits
give an in-control ARL near 30 only with the full d subtracted. The preprint
uses the full d (d = 0.5). This script scores the same runs with the printed
form (an allowance of d/2 = 0.25) next to the full d, so a reader can see how
much the choice moves the comparison: FedRR at its own control limit (the H
that gives ARL 30 for each allowance), and at the five-run empirical threshold
on the mid-severity population. Scoring only; no runs.

    python scripts/fedrr_sensitivity.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import review4_scores as r4  # noqa: E402
import score_faults as sf  # noqa: E402

ALLOWANCES = {"printed (d/2)": 0.25, "used (d)": 0.5}
STATS = ("fedrr", "fedrr_steps")


def native_rows(allowance: float) -> dict:
    """FedRR at its own control limit for one allowance."""
    sf.FEDRR_D = allowance
    families: dict = {}
    for f in sorted(sf.RESULTS.glob("probes_*.json")):
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
    faults = []
    for f in sorted(sf.RESULTS.glob("fault_*.json")):
        d, probs, upd = sf.load(f.stem)
        if (
            d["config"]["dataset"] != "fashion_mnist"
            or d.get("contact")
            or not sf.in_main_grid(d)
        ):
            continue
        faults.append((d, sf.scored(d, probs, upd)))
    summary: dict = {}
    with contextlib.redirect_stdout(io.StringIO()):
        sf.native_fedrr(families, faults, summary)
    native = summary["fedrr_native"]
    return {
        "h": native["h"],
        "rows": {
            (r["alpha"], r["stat"]): r
            for r in native["rows"]
            if r["stat"] in STATS
        },
    }


def pooled_rows(allowance: float, tmp: Path) -> dict:
    """Five-run empirical threshold, mid-severity population, main grid."""
    sf.FEDRR_D = allowance
    argv = sys.argv
    sys.argv = ["review4_scores.py", "--out", str(tmp)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r4.main()
    finally:
        sys.argv = argv
    return {
        (r["alpha"], r["stat"]): r
        for r in json.loads(tmp.read_text())
        if r["protocol"] == "pooled5"
        and r["stat"] in STATS
        and r["partition_seed"] == 0
        and r["clip_norm"] is None
    }


def main() -> int:
    """Score both allowances and write results/fedrr_sensitivity.json."""
    tmp = REPO / "results" / ".fedrr_sensitivity_tmp.json"
    out: dict = {}
    for label, d in ALLOWANCES.items():
        nat = native_rows(d)
        poo = pooled_rows(d, tmp)
        out[label] = {
            "allowance": d,
            "h": nat["h"],
            "native": [
                {"alpha": a, "stat": s, **{k: v for k, v in r.items()}}
                for (a, s), r in sorted(nat["rows"].items())
            ],
            "pooled5": [
                {
                    "alpha": a,
                    "stat": s,
                    "caught": r["mid"]["detected"],
                    "strict": r["mid"]["strict"],
                    "n": r["mid"]["n"],
                    "node_round_far": r["node_round_far"],
                }
                for (a, s), r in sorted(poo.items())
            ],
        }
    tmp.unlink(missing_ok=True)
    sf.FEDRR_D = 0.5
    dest = REPO / "results" / "fedrr_sensitivity.json"
    dest.write_text(json.dumps(out, indent=1))
    print(
        "| allowance | H | alpha | stat | own limit: alarmed runs | FAR % | "
        "caught | pooled5 caught (strict) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for label, blk in out.items():
        for nr in blk["native"]:
            pr = next(
                p
                for p in blk["pooled5"]
                if p["alpha"] == nr["alpha"] and p["stat"] == nr["stat"]
            )
            print(
                f"| {label} | {blk['h']:.2f} | {nr['alpha']} | {nr['stat']} | "
                f"{round(nr['runs_any_alarm'] * nr['healthy_runs'])}/"
                f"{nr['healthy_runs']} | {100 * nr['node_round_far']:.1f} | "
                f"{nr['detected']}/{nr['fault_runs']} | "
                f"{pr['caught']} ({pr['strict']}) of {pr['n']} |"
            )
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
