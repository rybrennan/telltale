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

The update-space baselines (telltale/baselines.py) need every run to carry a
.updates.npz sidecar. With --update-stats a run is done only when its sidecar
exists; an old run without one is re-run, its probe scores are checked bit for
bit against the stored file, and only the sidecar is written:

    nohup python scripts/run_grid.py --update-stats --seeds 0,1,2,3,4,5 \
        --jobs 3 > grid6.log 2>&1 &

--jobs runs that many runs at once. Each process keeps torch's default thread
count, so results do not depend on it (checked by the bitwise comparison).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
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
    seeds: list[int] = SEEDS,
    alphas: list[float] | None = None,
) -> list[tuple[str, list[str]]]:
    """List every run in the grid.

    Args:
        extra_nodes: Add EXTRA_NODES at the middle severity of each ladder.
        dataset: Dataset name passed through to run_probes.py.
        local_epochs: Local epochs per round, passed through.
        tag_prefix: Prefix that keeps a second modality's files distinct.
        seeds: Train seeds; the lowest calibrates, the rest are held out.
        alphas: Heterogeneity levels to include; None means all of ALPHAS.

    Returns:
        (expected result stem, argv) for every run, healthy run first per
        alpha and seed.
    """
    pcfg = ProbeConfig()
    runs = []
    for alpha, tag0 in ALPHAS.items():
        if alphas is not None and alpha not in alphas:
            continue
        tag = tag_prefix + tag0
        for seed in seeds:
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
    p.add_argument(
        "--seeds",
        default=",".join(map(str, SEEDS)),
        help="comma-separated train seeds",
    )
    p.add_argument(
        "--update-stats",
        action="store_true",
        help="record update-space baselines; a run is done when its "
        ".updates.npz exists",
    )
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument(
        "--alphas",
        default=None,
        help="comma-separated heterogeneity levels, e.g. 0.5,100 (default all)",
    )
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    runs = plan(
        extra_nodes=args.nodes,
        dataset=args.dataset,
        local_epochs=args.local_epochs,
        tag_prefix=args.tag_prefix,
        seeds=seeds,
        alphas=None
        if args.alphas is None
        else [float(a) for a in args.alphas.split(",")],
    )
    suffix = ".updates.npz" if args.update_stats else ".json"
    extra = ["--update-stats"] if args.update_stats else []
    todo = [
        (stem, argv + extra)
        for stem, argv in runs
        if not (RESULTS / f"{stem}{suffix}").exists()
    ]
    print(
        f"grid      {len(runs)} runs; {len(runs) - len(todo)} already done; "
        f"{len(todo)} to run"
    )
    print(
        f"priced    {len(todo) * args.minutes_per_run / 60:.1f} hours at "
        f"{args.minutes_per_run} min/run, sequential; --jobs {args.jobs}"
    )
    if args.dry_run:
        for _stem, argv in todo:
            print("  ", " ".join(argv))
        return 0

    started = time.time()
    # Healthy runs first: a fault run compares itself to its healthy run.
    todo.sort(key=lambda sa: not sa[0].startswith("probes_"))

    def one(i: int, argv: list[str]) -> int:
        print(
            f"\n===== [{i}/{len(todo)}] {' '.join(argv)}  "
            f"{time.strftime('%H:%M:%S')}",
            flush=True,
        )
        log = RESULTS.parent / "logs"
        log.mkdir(exist_ok=True)
        with open(log / f"run_{i:04d}.log", "w") as fh:
            rc = subprocess.call(
                [str(PY), str(REPO / "scripts" / "run_probes.py"), *argv],
                cwd=REPO,
                stdout=fh,
                stderr=subprocess.STDOUT,
            )
        status = "ok" if rc == 0 else f"FAILED rc={rc}"
        print(
            f"===== [{i}/{len(todo)}] {status}  {time.strftime('%H:%M:%S')}",
            flush=True,
        )
        return rc

    n_healthy = sum(s.startswith("probes_") for s, _ in todo)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(
            pool.map(
                one, range(1, n_healthy + 1), [a for _, a in todo[:n_healthy]]
            )
        )
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rcs = list(
            pool.map(
                one,
                range(n_healthy + 1, len(todo) + 1),
                [a for _, a in todo[n_healthy:]],
            )
        )
    print(f"\nfailed    {sum(rc != 0 for rc in rcs)} fault runs", flush=True)
    print(f"\nGRID_DONE  {(time.time() - started) / 3600:.1f} h", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
