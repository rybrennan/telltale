#!/usr/bin/env python3
"""Stages 2 and 3: one federation with the probe monitor attached.

Optionally, a fault is injected at one node.

Same federation as scripts/run_baseline.py plus a ProbeMonitor on the
update_hook seam scoring every client's returned state on a fixed battery.
With --kind, a FaultSpec is applied to --node through the fault seam. Outputs
land in results/:

    probes_<tag>_<fedfp>_<probefp>.npz
        healthy: probs[round, client, probe, class]
    probes_<tag>_<fedfp>_<probefp>.json
        config, accuracy curve, instrument check, characterisation per
        battery size
    fault_<tag>_<fedfp>_<probefp>_<faultfp>.*
        the same, plus the fault spec and the fleet-accuracy delta against
        the healthy run

A fault run reuses the healthy run's seeds so the only difference is the fault.

If the matching stage-1 baseline JSON exists, the instrumented accuracy curve
is compared against it and the result recorded. That is the at-scale form of
the inertness test: the monitor must not move a single number.

    python scripts/run_probes.py --alpha 0.5
    python scripts/run_probes.py --alpha 100 --tag iid
    python scripts/run_probes.py --alpha 0.1 --tag severe
    python scripts/run_probes.py --alpha 0.5 --kind bias --node 2 \
        --onset 15 --ramp 10 --severity 0.5

With --update-stats an UpdateStatsMonitor also records the update-space
signals the FedRR and FedCE baselines use, written to <stem>.updates.npz. If
the run's result files already exist (an earlier run of the same config),
they are not rewritten: the new probe scores are compared bit for bit with
the stored ones, the comparison is recorded in the sidecar, and only the
sidecar is written. Re-running an old seed for its update statistics is
therefore also a reproducibility check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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
from telltale.baselines import UpdateStatsMonitor  # noqa: E402
from telltale.contact import ContactSchedule  # noqa: E402
from telltale.detect import (  # noqa: E402
    characterise,
    check_reports,
    level_series,
    offset_change,
    persistence_series,
)
from telltale.faults import KINDS, FaultSpec, make_fault  # noqa: E402
from telltale.probes import (  # noqa: E402
    ProbeConfig,
    ProbeMonitor,
    build_probe_battery,
    model_bytes,
    payload_bytes,
)

RESULTS = REPO / "results"
SIZES = [10, 20, 50, 100, 200, 500]


def parse_args() -> argparse.Namespace:
    """Parse the probe run's command line into a namespace."""
    p = argparse.ArgumentParser(
        description="Healthy federation with probe monitor."
    )
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.5)
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
    p.add_argument("--n-probes", type=int, default=max(SIZES))
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--tag", default="baseline")
    p.add_argument(
        "--kind",
        choices=KINDS,
        default=None,
        help="inject a fault of this kind",
    )
    p.add_argument("--node", type=int, default=0)
    p.add_argument("--onset", type=int, default=15)
    p.add_argument("--ramp", type=int, default=0)
    p.add_argument("--severity", type=float, default=0.0)
    p.add_argument("--fault-seed", type=int, default=0)
    p.add_argument(
        "--sub-nodes",
        default="",
        help="comma-separated nodes on the submarine schedule",
    )
    p.add_argument(
        "--contact-period",
        type=int,
        default=1,
        help="submarine nodes report every k rounds",
    )
    p.add_argument(
        "--update-stats",
        action="store_true",
        help="also record update-space baseline signals (.updates.npz)",
    )
    return p.parse_args()


def main() -> int:
    """Run one monitored federation, check it, and write .json and .npz."""
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
    pcfg = ProbeConfig(n_probes=args.n_probes, probe_seed=args.probe_seed)
    sizes = [s for s in SIZES if s <= pcfg.n_probes] or [pcfg.n_probes]

    print(
        f"config    {cfg.tag}  fed={cfg.fingerprint()}  "
        f"probes={pcfg.fingerprint()}"
    )
    ds = load_dataset(cfg.dataset)
    parts = dirichlet_partition(
        ds.train_y,
        cfg.n_clients,
        cfg.dirichlet_alpha,
        cfg.partition_seed,
        cfg.min_client_samples,
    )
    summary = skew_summary(
        partition_label_matrix(ds.train_y, parts, ds.n_classes)
    )
    print(
        f"partition alpha={cfg.dirichlet_alpha}  mean TV "
        f"{summary['mean_tv_from_uniform']:.3f}"
    )

    battery = build_probe_battery(ds, pcfg)
    monitor = ProbeMonitor(battery, device=cfg.device)
    print(
        f"battery   {battery.n_probes} probes from {battery.source}  "
        f"fp={battery.fingerprint}"
    )

    schedule = None
    participation = None
    if args.sub_nodes:
        subs = tuple(int(x) for x in args.sub_nodes.split(","))
        schedule = ContactSchedule(period=args.contact_period, sub_nodes=subs)
        participation = lambda r: schedule.participants(r, cfg.n_clients)  # noqa: E731
        print(f"contact   {schedule}  fp={schedule.fingerprint()}")

    spec = None
    fault = None
    if args.kind is not None:
        spec = FaultSpec(
            args.kind,
            args.node,
            args.onset,
            args.severity,
            args.ramp,
            args.fault_seed,
        )
        fault = make_fault([spec], ds.n_classes)
        print(f"fault     {spec}  fp={spec.fingerprint()}")

    def report(rec) -> None:  # rec: telltale.fedavg.RoundRecord
        """Print one round's accuracy, loss and wall time."""
        print(
            f"  round {rec.round_idx + 1:>3d}/{cfg.rounds}  test_acc "
            f"{rec.test_acc:.4f}  "
            f"train_loss {rec.mean_train_loss:.4f}  {rec.seconds:.1f}s",
            flush=True,
        )

    stats = None
    hook = monitor
    if args.update_stats:
        stats = UpdateStatsMonitor(
            cfg.rounds,
            cfg.n_clients,
            ds.n_classes,
            cfg.init_seed,
            device=cfg.device,
        )

        def hook(updates, rnd) -> None:
            monitor(updates, rnd)
            stats(updates, rnd)

    t0 = time.perf_counter()
    result = run_federated(
        cfg,
        ds,
        parts,
        partition_summary=summary,
        on_round=report,
        update_hook=hook,
        fault=fault,
        participation=participation,
    )
    elapsed = time.perf_counter() - t0
    print(f"final     test_acc {result.final_acc:.4f}  {elapsed / 60:.1f} min")

    # ── against the stored healthy run: identity if no fault, delta if fault ──
    baseline_match = None
    baseline_file = next(
        RESULTS.glob(f"baseline_*_{cfg.fingerprint()}.json"), None
    )
    if spec is not None:
        suffix = f"_c{schedule.fingerprint()}" if schedule is not None else ""
        healthy_file = next(
            RESULTS.glob(
                f"probes_*_{cfg.fingerprint()}_{pcfg.fingerprint()}{suffix}.json"
            ),
            None,
        )
        if healthy_file is not None:
            healthy = json.loads(healthy_file.read_text())
            h_curve = [r["test_acc"] for r in healthy["rounds"]]
            tail = slice(max(0, cfg.rounds - 10), cfg.rounds)
            baseline_match = {
                "healthy_file": healthy_file.name,
                "healthy_tail_acc_mean": float(np.mean(h_curve[tail])),
                "faulted_tail_acc_mean": float(
                    np.mean(result.accuracy_curve()[tail])
                ),
                "pre_onset_curve_identical": h_curve[: spec.onset]
                == result.accuracy_curve()[: spec.onset],
            }
            print(f"healthy   {healthy_file.name}: {baseline_match}")
        else:
            print(
                "healthy   no stage-2 healthy run for these seeds; run it first"
            )
    elif baseline_file is not None:
        base = json.loads(baseline_file.read_text())
        base_curve = [r["test_acc"] for r in base["rounds"]]
        base_loss = [r["mean_train_loss"] for r in base["rounds"]]
        baseline_match = {
            "file": baseline_file.name,
            "accuracy_curve_identical": base_curve == result.accuracy_curve(),
            "train_loss_identical": base_loss
            == [r.mean_train_loss for r in result.rounds],
        }
        print(f"baseline  {baseline_file.name}: {baseline_match}")
    else:
        print(
            "baseline  no stage-1 file for this fingerprint; inertness not "
            "checked at scale"
        )

    # ── instrument self-check over every round ──
    issues: list[str] = []
    prev: dict[int, torch.Tensor] = {}
    for _rnd, reps in sorted(monitor.by_round().items()):
        issues += check_reports(reps, battery, previous=prev)
        prev = {r.client_id: r.probs for r in reps}
    print(
        f"instrument {len(issues)} issues"
        + ("" if not issues else ": " + "; ".join(issues[:5]))
    )

    probs = monitor.probs_array(cfg.rounds, cfg.n_clients)
    rows = characterise(probs, sizes=sizes)

    per_node = None
    if spec is not None:
        lv = level_series(probs)
        ps = persistence_series(offset_change(lv["offset"], 5), 5)
        tail = slice(max(0, cfg.rounds - 10), cfg.rounds)
        per_node = {
            "level_tail_mean": np.nanmean(lv["level"][tail], axis=0).tolist(),
            "z_tail_max": np.nanmax(np.abs(lv["z"][tail]), axis=0).tolist(),
            "persist_tail_max": np.nanmax(ps[tail], axis=0).tolist(),
            "persist_by_round_target": ps[:, spec.node].tolist(),
        }
        print(f"\n{'node':>5} {'level':>8} {'z_max':>6} {'persist_max':>12}")
        for k in range(cfg.n_clients):
            mark = "  <- fault" if k == spec.node else ""
            print(
                f"{k:>5} {per_node['level_tail_mean'][k]:>8.4f} "
                f"{per_node['z_tail_max'][k]:>6.2f} "
                f"{per_node['persist_tail_max'][k]:>12.3f}{mark}"
            )
    mbytes = model_bytes(result.final_state)
    for r in rows:
        r["payload_bytes_float16"] = payload_bytes(
            r["n_probes"], ds.n_classes, "float16"
        )
        r["payload_fraction_of_model"] = r["payload_bytes_float16"] / mbytes

    print(
        f"\n{'n':>5} {'bytes':>7} {'level_med':>10} {'round_sd':>9} "
        f"{'client_sd':>10} {'z_max':>6} {'self_med':>9} {'persist':>8}"
    )
    for r in rows:
        print(
            f"{r['n_probes']:>5} {r['payload_bytes_float16']:>7} "
            f"{r['level_median']:>10.5f} "
            f"{r['level_round_sd']:>9.5f} {r['level_client_sd']:>10.5f} "
            f"{r['level_z_max']:>6.2f} "
            f"{r['self_median']:>9.5f} {r['persist_median']:>8.3f}"
        )

    RESULTS.mkdir(exist_ok=True)
    stem = f"probes_{cfg.tag}_{cfg.fingerprint()}_{pcfg.fingerprint()}"
    if spec is not None:
        stem = (
            f"fault_{cfg.tag}_{cfg.fingerprint()}_{pcfg.fingerprint()}_"
            f"{spec.fingerprint()}"
        )
    if schedule is not None:
        stem += f"_c{schedule.fingerprint()}"
    existing = RESULTS / f"{stem}.npz"
    if stats is not None and existing.exists():
        stored = np.load(existing)["probs"]
        match = bool(
            stored.shape == probs.shape
            and np.array_equal(stored, probs, equal_nan=True)
        )
        np.savez_compressed(
            RESULTS / f"{stem}.updates.npz",
            reproduction=np.array("identical" if match else "differ"),
            **stats.arrays(),
        )
        print(
            f"existing  results/{stem}.* kept; probe scores "
            f"{'IDENTICAL' if match else 'DIFFER'} to the stored run"
        )
        print(f"wrote     results/{stem}.updates.npz  q={stats.q}")
        return 0 if match else 3
    if stats is not None:
        np.savez_compressed(
            RESULTS / f"{stem}.updates.npz",
            reproduction=np.array("new"),
            **stats.arrays(),
        )
        print(f"wrote     results/{stem}.updates.npz  q={stats.q}")
    np.savez_compressed(
        RESULTS / f"{stem}.npz",
        probs=probs,
        battery_indices=battery.indices,
        battery_labels=battery.y.numpy(),
    )
    (RESULTS / f"{stem}.json").write_text(
        json.dumps(
            {
                "config": cfg.to_dict(),
                "fed_fingerprint": cfg.fingerprint(),
                "probe_config": pcfg.to_dict(),
                "probe_fingerprint": pcfg.fingerprint(),
                "battery_fingerprint": battery.fingerprint,
                "partition_summary": summary,
                "rounds": [r.to_dict() for r in result.rounds],
                "final_acc": result.final_acc,
                "elapsed_seconds": elapsed,
                "model_bytes": mbytes,
                "baseline_match": baseline_match,
                "instrument_issues": issues,
                "characterisation": rows,
                "fault": spec.to_dict() if spec is not None else None,
                "fault_fingerprint": spec.fingerprint()
                if spec is not None
                else None,
                "per_node": per_node,
                "contact": schedule.to_dict() if schedule is not None else None,
            },
            indent=2,
        )
    )
    print(f"wrote     results/{stem}.json + .npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
