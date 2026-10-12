#!/usr/bin/env python3
"""The integrity layer running inside a real Flower federation.

A Flower server (gRPC) and N client processes on this machine. Each client
trains locally, scores the shared probe battery with its post-training model,
and returns the score vector in Flower's metrics channel. The server-side
strategy (`IntegrityFedAvg`) runs the harness's own detector on those vectors.

Protocol mirrors the stage-4 grid: 30 rounds, CUSUM reference epoch rounds
5-14, one node's feed faulted from round 15 with a 10-round ramp, threshold
set on a healthy run.

    python scripts/flower_demo.py               # healthy run, then faulted run
    python scripts/flower_demo.py --quarantine  # faulted run drops flagged

This demonstrates integration, not tactical deployment: every process is on
one host, over loopback. It is the ICD exercised end to end.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

import os

import numpy as np  # noqa: E402
import torch  # noqa: E402

from telltale.config import FedConfig  # noqa: E402
from telltale.data import dirichlet_partition, load_dataset  # noqa: E402
from telltale.faults import FaultSpec, make_fault  # noqa: E402
from telltale.fedavg import evaluate, local_train  # noqa: E402
from telltale.models import build_model  # noqa: E402
from telltale.probes import (  # noqa: E402
    ProbeConfig,
    build_probe_battery,
    score_probes,
)

N_NODES, ALPHA = 5, 0.5
ROUNDS = int(os.environ.get("TELLTALE_ROUNDS", "30"))
ONSET, RAMP, REFERENCE = 15, 10, slice(5, 15)
N_PROBES = 50  # ~1 kB at float16 x 10 classes: the brief's operating point


def free_port() -> int:
    """Return a TCP port on loopback that is free right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── client process ───────────────────────────────────────────────────────────
def run_client(node: int, address: str, fault_json: str | None) -> None:
    """Run one Flower client process until the server ends the federation.

    The client trains on its own Dirichlet shard, scores the probe battery
    with its post-training model and ships the scores in the fit metrics.

    Args:
        node: Index of this node in the federation.
        address: Server address as ``host:port``.
        fault_json: JSON-encoded FaultSpec for this node, or None if healthy.
    """
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)
    import flwr
    from flwr.compat.client.app import start_client

    from telltale.integrations.flower import (
        FINGERPRINT_KEY,
        NODE_KEY,
        PROBE_KEY,
        encode_scores,
    )

    cfg = FedConfig(n_clients=N_NODES, rounds=ROUNDS, dirichlet_alpha=ALPHA)
    ds = load_dataset(cfg.dataset)
    parts = dirichlet_partition(
        ds.train_y,
        n_clients=N_NODES,
        alpha=ALPHA,
        seed=cfg.partition_seed,
        min_client_samples=cfg.min_client_samples,
    )
    idx = torch.as_tensor(parts[node])
    x0, y0 = ds.train_x[idx], ds.train_y[idx]
    battery = build_probe_battery(ds, ProbeConfig(n_probes=N_PROBES))
    fault = (
        make_fault([FaultSpec(**json.loads(fault_json))], ds.n_classes)
        if fault_json
        else None
    )
    model = build_model("small_cnn", ds.n_classes, seed=cfg.init_seed)
    keys = list(model.state_dict().keys())

    class Node(flwr.client.NumPyClient):
        def get_parameters(self, config: dict) -> list[np.ndarray]:
            return [
                v.detach().cpu().numpy() for v in model.state_dict().values()
            ]

        def fit(
            self, parameters: list[np.ndarray], config: dict
        ) -> tuple[list[np.ndarray], int, dict]:
            rnd = int(config.get("round", 1)) - 1
            # Server weights come from the same small_cnn, so the lengths
            # match by construction.
            model.load_state_dict(
                {
                    k: torch.as_tensor(v)
                    for k, v in zip(keys, parameters, strict=False)
                }
            )
            x, y = fault(node, rnd, x0, y0) if fault else (x0, y0)
            state, loss = local_train(
                model,
                x,
                y,
                cfg,
                seed=hash((cfg.train_seed, rnd, node)) & 0x7FFFFFFF,
            )
            model.load_state_dict(state)
            # Scored ON the node, before anything leaves it.
            probs = score_probes(model, battery).detach().numpy()
            return (
                self.get_parameters({}),
                int(x.shape[0]),
                {
                    NODE_KEY: node,
                    PROBE_KEY: encode_scores(probs),
                    FINGERPRINT_KEY: battery.fingerprint,
                    "train_loss": float(loss),
                },
            )

        def evaluate(
            self, parameters: list[np.ndarray], config: dict
        ) -> tuple[float, int, dict]:
            return 0.0, 1, {}

    start_client(
        server_address=address, client=Node().to_client(), insecure=True
    )


# ── server ───────────────────────────────────────────────────────────────────
def run_federation(
    fault: FaultSpec | None, threshold: float | None, quarantine: bool
) -> dict:
    """Run one in-process Flower server with N_NODES spawned clients.

    Args:
        fault: Fault to inject on ``fault.node``, or None for a healthy run.
        threshold: CUSUM alarm threshold, or None when calibrating.
        quarantine: Whether the strategy excludes flagged nodes.

    Returns:
        A dict with the CUSUM matrix (None if the run is too short for the
        reference epoch), the strategy state, per-round fleet accuracy and
        the wall-clock seconds.
    """
    import flwr
    from flwr.server import ServerConfig

    from telltale.integrations.flower import IntegrityFedAvg

    ds = load_dataset("fashion_mnist")
    battery = build_probe_battery(ds, ProbeConfig(n_probes=N_PROBES))
    test_x, test_y = ds.test_x, ds.test_y
    eval_model = build_model("small_cnn", ds.n_classes, seed=0)
    keys = list(eval_model.state_dict().keys())
    fleet_acc: list[float] = []

    def central_eval(
        server_round: int, parameters: list[np.ndarray], config: dict
    ) -> tuple[float, dict]:
        """Score the global model on the test set after each round."""
        # Same small_cnn on both sides, so the lengths match by construction.
        eval_model.load_state_dict(
            {
                k: torch.as_tensor(v)
                for k, v in zip(keys, parameters, strict=False)
            }
        )
        loss, acc = evaluate(eval_model, test_x, test_y)
        if server_round > 0:
            fleet_acc.append(acc)
        return loss, {"accuracy": acc}

    strategy = IntegrityFedAvg(
        n_nodes=N_NODES,
        n_probes=N_PROBES,
        n_classes=ds.n_classes,
        battery_fingerprint=battery.fingerprint,
        n_rounds=ROUNDS,
        reference=REFERENCE,
        threshold=threshold,
        quarantine=quarantine,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=N_NODES,
        min_available_clients=N_NODES,
        on_fit_config_fn=lambda rnd: {"round": rnd},
        evaluate_fn=central_eval,
    )
    port = free_port()
    address = f"127.0.0.1:{port}"
    ctx = mp.get_context("spawn")
    fj = json.dumps(fault.to_dict()) if fault else None
    procs = [
        ctx.Process(
            target=run_client,
            args=(k, address, fj if fault and k == fault.node else None),
        )
        for k in range(N_NODES)
    ]
    t0 = time.time()
    import threading

    def launch() -> None:
        """Start the clients once the server has had time to bind."""
        time.sleep(2.0)
        for p in procs:
            p.start()

    threading.Thread(target=launch, daemon=True).start()
    flwr.server.start_server(
        server_address=address,
        config=ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
    )
    for p in procs:
        p.join(timeout=30)
    c = strategy.cusum_so_far() if ROUNDS >= REFERENCE.stop else None
    return {
        "cusum": c,
        "state": strategy.state,
        "fleet_acc": fleet_acc,
        "seconds": time.time() - t0,
    }


def main() -> int:
    """Calibrate on a healthy run, then score one faulted run."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--quarantine", action="store_true")
    ap.add_argument("--kind", default="blur")
    ap.add_argument("--severity", type=float, default=1.5)
    ap.add_argument("--node", type=int, default=2)
    args = ap.parse_args()

    print(f"[healthy] {N_NODES} Flower clients, {ROUNDS} rounds, alpha={ALPHA}")
    h = run_federation(None, None, False)
    post = h["cusum"][REFERENCE.stop :]
    thr = float(np.nanmax(post))
    print(
        f"[healthy] done in {h['seconds']:.0f}s; "
        f"CUSUM healthy max after reference = {thr:.2f} -> threshold"
    )

    spec = FaultSpec(args.kind, args.node, ONSET, args.severity, RAMP)
    print(
        f"[faulted] {args.kind} severity {args.severity} on node "
        f"{args.node}, onset {ONSET}, ramp {RAMP}"
        f"{', QUARANTINE on' if args.quarantine else ''}"
    )
    f = run_federation(spec, thr, args.quarantine)
    c, st = f["cusum"], f["state"]

    flag_rounds = {
        k: [r for r in range(ROUNDS) if np.isfinite(c[r, k]) and c[r, k] > thr]
        for k in range(N_NODES)
    }
    first = {k: (v[0] if v else None) for k, v in flag_rounds.items()}
    hit = first[args.node]
    false_nodes = [
        k for k, v in first.items() if v is not None and k != args.node
    ]
    h_tail, f_tail = np.mean(h["fleet_acc"][-5:]), np.mean(f["fleet_acc"][-5:])

    summary = {
        "framework": "flwr " + __import__("flwr").__version__,
        "nodes": N_NODES,
        "rounds": ROUNDS,
        "alpha": ALPHA,
        "fault": spec.to_dict(),
        "quarantine": args.quarantine,
        "threshold_cusum_healthy_max": thr,
        "detected": hit is not None,
        "first_flag_round": hit,
        "rounds_after_onset": (hit - ONSET) if hit is not None else None,
        "falsely_flagged_nodes": false_nodes,
        "payload_bytes_per_node_round": int(np.median(st.payload_bytes)),
        "fingerprint_mismatches": len(st.fingerprint_mismatches),
        "fleet_acc_tail_healthy": h_tail,
        "fleet_acc_tail_faulted": f_tail,
        "fleet_acc_tail_delta": f_tail - h_tail,
        "quarantined_from_round": min(st.quarantined)
        if st.quarantined
        else None,
    }
    out = (
        REPO
        / "results"
        / f"flower_demo_{args.kind}_{'q' if args.quarantine else 'noq'}.json"
    )
    out.write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
