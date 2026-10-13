r"""Run FedAvg with the telltale integrity aggregator in an NVFlare simulation.

    python job.py                        # healthy: 10 sites, 30 rounds
    python job.py --participation 0.6    # 6 of 10 sites sampled each round
    python job.py --fault_site 3 --fault_kind blur --fault_severity 1.5 \\
        --threshold 3.8                  # site-3 blurs from round 15

The aggregator writes its full detector history (every round's probe scores,
the CUSUM path, flags, participants) to `--history_path`.
`scripts/nvflare_eval.py` drives this job across heterogeneity and
participation levels and scores the histories.
"""

from __future__ import annotations

import argparse
import os

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv

from telltale.data import load_dataset
from telltale.integrations.nvflare import IntegrityAggregator
from telltale.models import build_model
from telltale.probes import ProbeConfig, build_probe_battery

HERE = os.path.dirname(os.path.abspath(__file__))


def define_parser() -> argparse.ArgumentParser:
    """Build the job's argument parser.

    Returns:
        The parser for the federation, detector and fault settings.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = parser.add_argument
    add("--n_clients", type=int, default=10, help="Sites (default: 10).")
    add("--num_rounds", type=int, default=30, help="Rounds (default: 30).")
    add(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet label-skew concentration (default: 0.5).",
    )
    add(
        "--participation",
        type=float,
        default=1.0,
        help="Fraction of sites sampled each round (default: 1.0).",
    )
    add("--train_seed", type=int, default=0)
    add(
        "--n_probes",
        type=int,
        default=50,
        help="Probe battery size; 50 probes x 10 classes is about 1 kB.",
    )
    add(
        "--reference_start",
        type=int,
        default=5,
        help="First round of the CUSUM reference epoch.",
    )
    add(
        "--reference_stop",
        type=int,
        default=15,
        help="Round after the last round of the CUSUM reference epoch.",
    )
    add(
        "--threshold",
        type=float,
        default=None,
        help="CUSUM threshold; unset records without flagging.",
    )
    add(
        "--exclude_flagged",
        action="store_true",
        help="Leave flagged sites out of the average (off by default).",
    )
    add(
        "--fault_site",
        type=int,
        default=0,
        help="1-based site to fault; 0 for a healthy run.",
    )
    add(
        "--fault_kind",
        type=str,
        default="blur",
        help="bias, gain, blur or label_noise.",
    )
    add("--fault_severity", type=float, default=1.5)
    add(
        "--fault_onset",
        type=int,
        default=15,
        help="First faulted round (0-based).",
    )
    add("--fault_ramp", type=int, default=10)
    add(
        "--history_path",
        type=str,
        default="/tmp/nvflare/telltale/history.json",
    )
    add("--name", type=str, default="telltale-integrity")
    return parser


def create_recipe(args: argparse.Namespace) -> FedAvgRecipe:
    """Build the FedAvg recipe with the integrity aggregator attached.

    Args:
        args: Parsed `define_parser` arguments.

    Returns:
        A recipe ready to execute in a simulation environment.
    """
    ds = load_dataset("fashion_mnist")
    battery = build_probe_battery(ds, ProbeConfig(n_probes=args.n_probes))
    sampled = max(2, round(args.participation * args.n_clients))

    aggregator = IntegrityAggregator(
        n_probes=args.n_probes,
        n_classes=ds.n_classes,
        battery_fingerprint=battery.fingerprint,
        num_rounds=args.num_rounds,
        max_clients=args.n_clients,
        reference_start=args.reference_start,
        reference_stop=args.reference_stop,
        threshold=args.threshold,
        exclude_flagged=args.exclude_flagged,
        history_path=args.history_path,
    )
    train_args = (
        f"--n_clients {args.n_clients} --alpha {args.alpha}"
        f" --train_seed {args.train_seed} --n_probes {args.n_probes}"
    )
    fault_args = (
        f" --fault_kind {args.fault_kind}"
        f" --fault_severity {args.fault_severity}"
        f" --fault_onset {args.fault_onset} --fault_ramp {args.fault_ramp}"
    )
    # Every site is listed so exactly one of them carries the fault.
    per_site_config = {
        f"site-{k}": {
            "train_args": train_args
            + (fault_args if k == args.fault_site else "")
        }
        for k in range(1, args.n_clients + 1)
    }

    return FedAvgRecipe(
        name=args.name,
        # FedAvg samples `min_clients` of the connected sites each round, so
        # a value below n_clients is partial participation.
        min_clients=sampled,
        num_rounds=args.num_rounds,
        model=build_model("small_cnn", ds.n_classes, seed=0),
        train_script=os.path.join(HERE, "client.py"),
        train_args=train_args,
        per_site_config=per_site_config,
        aggregator=aggregator,
    )


def main(argv: list[str] | None = None) -> None:
    """Run one simulated federation and report where the history went.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    args = define_parser().parse_args(argv)
    recipe = create_recipe(args)
    # Sites are named explicitly because each has its own per-site entry.
    sites = [f"site-{k}" for k in range(1, args.n_clients + 1)]
    run = recipe.execute(SimEnv(clients=sites))
    print("Job status:", run.get_status())
    print("Detector history:", args.history_path)


if __name__ == "__main__":
    main()
