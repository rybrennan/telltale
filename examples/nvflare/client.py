"""NVFlare client for the telltale integrity example.

Each site trains the harness's small CNN on its Dirichlet shard of
Fashion-MNIST, scores the shared probe battery with its post-training model,
and returns the scores in `FLModel.meta` next to its weights. Training,
partitioning and fault injection are the harness's own functions, so a run
here is the same experiment as a run in `scripts/run_grid.py`, carried by
NVFlare instead of the in-process loop.

Site names are `site-1` ... `site-N`; site `k` holds shard `k - 1`.
"""

from __future__ import annotations

import argparse

import nvflare.client as flare
import torch
from nvflare.apis.fl_constant import FLMetaKey

from telltale.config import FedConfig
from telltale.data import dirichlet_partition, load_dataset
from telltale.faults import FaultSpec, make_fault
from telltale.fedavg import local_train
from telltale.integrations.nvflare import probe_meta
from telltale.models import build_model
from telltale.probes import ProbeConfig, build_probe_battery, score_probes


def define_parser() -> argparse.ArgumentParser:
    """Build the client's argument parser.

    Returns:
        The parser for the arguments job.py passes to every site.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_clients", type=int, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Dirichlet concentration of the label split.",
    )
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--n_probes", type=int, default=50)
    parser.add_argument(
        "--fault_kind",
        type=str,
        default="",
        help="bias, gain, blur or label_noise; empty for a healthy site.",
    )
    parser.add_argument(
        "--fault_onset",
        type=int,
        default=15,
        help="First faulted round (0-based).",
    )
    parser.add_argument("--fault_severity", type=float, default=0.0)
    parser.add_argument(
        "--fault_ramp",
        type=int,
        default=10,
        help="Rounds over which severity ramps up.",
    )
    return parser


def site_index(site_name: str) -> int:
    """Map a site name to its shard index.

    Args:
        site_name: An NVFlare site name such as `site-3`.

    Returns:
        The 0-based shard index, so `site-3` gives 2.
    """
    return int(site_name.rsplit("-", 1)[1]) - 1


def main(argv: list[str] | None = None) -> None:
    """Train, score the probe battery, and report, every round.

    Args:
        argv: Command-line arguments; None reads `sys.argv`.
    """
    args = define_parser().parse_args(argv)
    torch.set_num_threads(1)

    flare.init()
    site = flare.get_site_name()
    node = site_index(site)

    cfg = FedConfig(
        n_clients=args.n_clients,
        dirichlet_alpha=args.alpha,
        train_seed=args.train_seed,
    )
    ds = load_dataset(cfg.dataset)
    parts = dirichlet_partition(
        ds.train_y,
        n_clients=cfg.n_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.partition_seed,
        min_client_samples=cfg.min_client_samples,
    )
    idx = torch.as_tensor(parts[node])
    x0, y0 = ds.train_x[idx], ds.train_y[idx]
    battery = build_probe_battery(ds, ProbeConfig(n_probes=args.n_probes))
    fault = None
    if args.fault_kind:
        spec = FaultSpec(
            args.fault_kind,
            node,
            args.fault_onset,
            args.fault_severity,
            args.fault_ramp,
        )
        fault = make_fault([spec], ds.n_classes)
    model = build_model("small_cnn", ds.n_classes, seed=cfg.init_seed)

    while flare.is_running():
        input_model = flare.receive()
        rnd = input_model.current_round
        model.load_state_dict(
            {k: torch.as_tensor(v) for k, v in input_model.params.items()}
        )

        x, y = fault(node, rnd, x0, y0) if fault else (x0, y0)
        seed = hash((cfg.train_seed, rnd, node)) & 0x7FFFFFFF
        state, loss = local_train(model, x, y, cfg, seed=seed)
        model.load_state_dict(state)
        # Scored on the client; the server never runs a probe forward pass.
        probs = score_probes(model, battery).detach().numpy()

        meta = {FLMetaKey.NUM_STEPS_CURRENT_ROUND: int(x.shape[0])}
        meta.update(probe_meta(probs, battery.fingerprint))
        params = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        flare.send(
            flare.FLModel(
                params=params,
                metrics={"train_loss": float(loss)},
                meta=meta,
            )
        )


if __name__ == "__main__":
    main()
