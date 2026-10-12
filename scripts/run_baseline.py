#!/usr/bin/env python3
"""Step 1: a clean federation, no faults, no detector.

Establishes what normal looks like. Every later claim about detecting a
corrupted node is a claim about departure from this baseline, so the baseline
has to be characterised first, including how much run-to-run and
client-to-client variation exists when nothing is wrong at all.

    python scripts/run_baseline.py --rounds 30 --clients 10 --alpha 0.5
    python scripts/run_baseline.py --alpha 100 --tag iid      # near-IID control
    python scripts/run_baseline.py --alpha 0.1 --tag severe   # severe skew
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from telltale import (  # noqa: E402
    FedConfig,
    dirichlet_partition,
    load_dataset,
    partition_label_matrix,
    run_federated,
    skew_summary,
)

RESULTS = REPO / "results"


def parse_args() -> argparse.Namespace:
    """Parse the baseline's command line into a namespace."""
    p = argparse.ArgumentParser(description="Clean FedAvg baseline.")
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration"
    )
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=["fashion_mnist", "mnist", "deepship"],
    )
    p.add_argument("--partition-seed", type=int, default=0)
    p.add_argument("--init-seed", type=int, default=0)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--tag", default="baseline")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> int:
    """Train one clean federation, print progress and write the result."""
    args = parse_args()
    cfg = FedConfig(
        n_clients=args.clients,
        rounds=args.rounds,
        dirichlet_alpha=args.alpha,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset=args.dataset,
        partition_seed=args.partition_seed,
        init_seed=args.init_seed,
        train_seed=args.train_seed,
        tag=args.tag,
    )

    print(f"config    {cfg.tag}  fingerprint={cfg.fingerprint()}")
    print(f"loading   {cfg.dataset} ...")
    ds = load_dataset(cfg.dataset)
    print(
        f"          train={ds.n_train}  test={ds.test_x.shape[0]}  "
        f"classes={ds.n_classes}"
    )

    parts = dirichlet_partition(
        ds.train_y,
        n_clients=cfg.n_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.partition_seed,
        min_client_samples=cfg.min_client_samples,
    )
    matrix = partition_label_matrix(ds.train_y, parts, ds.n_classes)
    summary = skew_summary(matrix)
    print(
        f"partition {cfg.n_clients} clients @ alpha={cfg.dirichlet_alpha}  "
        f"sizes {summary['min_client_size']}-{summary['max_client_size']}  "
        f"mean TV from uniform {summary['mean_tv_from_uniform']:.3f}"
    )
    if not args.quiet:
        print("\n          per-client label counts")
        header = "          cid " + " ".join(
            f"{k:>5d}" for k in range(ds.n_classes)
        )
        print(header + "    total")
        for i, row in enumerate(matrix):
            print(
                f"          {i:>3d} "
                + " ".join(f"{c:>5d}" for c in row)
                + f"  {row.sum():>7d}"
            )

    print(f"\ntraining  {cfg.rounds} rounds, {cfg.n_clients} clients/round")

    def report(rec) -> None:  # rec: telltale.fedavg.RoundRecord
        """Print one round's accuracy, loss and wall time."""
        print(
            f"  round {rec.round_idx + 1:>3d}/{cfg.rounds}  "
            f"test_acc {rec.test_acc:.4f}  test_loss {rec.test_loss:.4f}  "
            f"train_loss {rec.mean_train_loss:.4f}  {rec.seconds:.1f}s"
        )

    result = run_federated(
        cfg, ds, parts, partition_summary=summary, on_round=report
    )

    print(
        f"\nfinal     test_acc {result.final_acc:.4f}  params "
        f"{result.n_parameters:,}"
    )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"baseline_{cfg.tag}_{cfg.fingerprint()}.json"
    out.write_text(json.dumps(result.to_dict(), indent=2))
    print(f"wrote     {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
