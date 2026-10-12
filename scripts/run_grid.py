#!/usr/bin/env python3
"""Stage 4 grid: healthy seeds and faults across alpha, kind, severity, seed.

Meant to be started from a terminal and left overnight, not run inside a
session. Idempotent: a run whose result file already exists is skipped, so
the grid can be stopped and resumed. Runs are sequential on purpose: the
thread count must match the stage-1 baseline for bitwise comparisons.

    python scripts/run_grid.py --dry-run     # count and price, launch nothing
    nohup python scripts/run_grid.py > grid.log 2>&1 &
    python scripts/run_grid.py --dataset deepship --local-epochs 5 \
        --tag-prefix ds-                     # acoustic arm

Then, in the morning:

    python scripts/score_faults.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from telltale import FedConfig  # noqa: E402
from telltale.faults import FaultSpec  # noqa: E402
from telltale.probes import ProbeConfig  # noqa: E402

RESULTS = REPO / "results"
PY = REPO / ".venv" / "bin" / "python"
MINUTES_PER_RUN = 6.6

ALPHAS = {0.1: "severe", 0.5: "baseline", 100.0: "iid"}
SEEDS = [0, 1, 2]
NODE, ONSET, RAMP = 2, 15, 10
# --nodes adds a node dimension at the middle severity of each ladder only, so
# the heterogeneity-vs-node-weight confound can be separated without tripling
# the grid.
EXTRA_NODES = [5, 9]
# Severity ladders per kind. Bias is flat by measurement
# (docs/stage3_fault_design.md s.2), so it carries one point as the weak
# control.
LADDER = {
    "label_noise": [0.1, 0.3, 0.5],
    "blur": [0.5, 1.0, 1.5],
    "gain": [0.25, 0.5, 1.0],
    "bias": [0.5],
}


def plan(
    extra_nodes: bool = False,
    dataset: str = "fashion_mnist",
    local_epochs: int = 1,
    tag_prefix: str = "",
) -> list[tuple[str, list[str]]]:
    """List every run in the grid.

    Args:
        extra_nodes: Add EXTRA_NODES at the middle severity of each ladder.
        dataset: Dataset name passed through to run_probes.py.
        local_epochs: Local epochs per round, passed through.
        tag_prefix: Prefix that keeps a second modality's files distinct.

    Returns:
        (expected result stem, argv) for every run, healthy run first per
        alpha and seed.
    """
    pcfg = ProbeConfig()
    runs = []
    for alpha, tag0 in ALPHAS.items():
        tag = tag_prefix + tag0
        for seed in SEEDS:
            cfg = FedConfig(
                dirichlet_alpha=alpha,
                tag=tag,
                train_seed=seed,
                dataset=dataset,
                local_epochs=local_epochs,
            )
            base = [
                "--alpha",
                str(alpha),
                "--tag",
                tag,
                "--train-seed",
                str(seed),
                "--dataset",
                dataset,
                "--local-epochs",
                str(local_epochs),
            ]
            runs.append(
                (f"probes_{tag}_{cfg.fingerprint()}_{pcfg.fingerprint()}", base)
            )
            for kind, sevs in LADDER.items():
                points = [(NODE, sev) for sev in sevs]
                if extra_nodes:
                    mid = sevs[len(sevs) // 2]
                    points += [(n, mid) for n in EXTRA_NODES]
                for node, sev in points:
                    spec = FaultSpec(kind, node, ONSET, sev, RAMP)
                    stem = (
                        f"fault_{tag}_{cfg.fingerprint()}_"
                        f"{pcfg.fingerprint()}_{spec.fingerprint()}"
                    )
                    runs.append(
                        (
                            stem,
                            base
                            + [
                                "--kind",
                                kind,
                                "--node",
                                str(node),
                                "--onset",
                                str(ONSET),
                                "--ramp",
                                str(RAMP),
                                "--severity",
                                str(sev),
                            ],
                        )
                    )
    return runs


def main() -> int:
    """Price the grid, then run every missing result in sequence."""
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--nodes",
        action="store_true",
        help="add the node dimension (extra nodes at mid severity)",
    )
    p.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=["fashion_mnist", "mnist", "deepship"],
    )
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument(
        "--tag-prefix",
        default="",
        help="e.g. 'ds-' so result files of a second modality are distinct",
    )
    p.add_argument("--minutes-per-run", type=float, default=MINUTES_PER_RUN)
    args = p.parse_args()

    runs = plan(
        extra_nodes=args.nodes,
        dataset=args.dataset,
        local_epochs=args.local_epochs,
        tag_prefix=args.tag_prefix,
    )
    todo = [
        (stem, argv)
        for stem, argv in runs
        if not (RESULTS / f"{stem}.json").exists()
    ]
    print(
        f"grid      {len(runs)} runs; {len(runs) - len(todo)} already done; "
        f"{len(todo)} to run"
    )
    print(
        f"priced    {len(todo) * args.minutes_per_run / 60:.1f} hours at "
        f"{args.minutes_per_run} min/run, sequential"
    )
    if args.dry_run:
        for _stem, argv in todo:
            print("  ", " ".join(argv))
        return 0

    started = time.time()
    for i, (_stem, argv) in enumerate(todo, 1):
        print(
            f"\n===== [{i}/{len(todo)}] {' '.join(argv)}  "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        rc = subprocess.call(
            [str(PY), str(REPO / "scripts" / "run_probes.py"), *argv], cwd=REPO
        )
        if rc != 0:
            print(f"===== run failed rc={rc}; continuing", flush=True)
    print(f"\nGRID_DONE  {(time.time() - started) / 3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
