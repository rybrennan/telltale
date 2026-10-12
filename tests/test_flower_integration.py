"""The integrity layer as a Flower strategy: no network, no Flower server.

`aggregate_fit` is driven directly with FitRes objects shaped exactly as the
client processes in scripts/flower_demo.py produce them.
"""

from __future__ import annotations

import numpy as np
import pytest

flwr = pytest.importorskip("flwr")
from flwr.common import (  # noqa: E402
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from telltale.integrations.flower import (  # noqa: E402
    FINGERPRINT_KEY,
    NODE_KEY,
    PROBE_KEY,
    IntegrityFedAvg,
    decode_scores,
    encode_scores,
)

K, P, C, R = 5, 20, 10, 30
FP = "battery-abc"


def fitres(
    node: int, probs: np.ndarray, weight: float, fp: str = FP
) -> tuple[None, FitRes]:
    """Build one client's fit result: constant weights plus a probe payload."""
    params = ndarrays_to_parameters([np.full((3,), weight, dtype=np.float32)])
    metrics = {
        NODE_KEY: node,
        PROBE_KEY: encode_scores(probs),
        FINGERPRINT_KEY: fp,
    }
    return None, FitRes(Status(Code.OK, ""), params, 100, metrics)


def strategy(**kw) -> IntegrityFedAvg:
    """Build the strategy under test with the module's fleet constants."""
    return IntegrityFedAvg(
        n_nodes=K,
        n_probes=P,
        n_classes=C,
        battery_fingerprint=FP,
        n_rounds=R,
        reference=slice(5, 15),
        fraction_evaluate=0.0,
        **kw,
    )


def healthy_probs(rng: np.random.Generator) -> np.ndarray:
    """Each node has its own stable class preference (heterogeneity).

    Per-round noise is added later by round_probs.
    """
    base = rng.dirichlet(np.ones(C), size=(K, P))
    return base


def round_probs(
    base: np.ndarray,
    rng: np.random.Generator,
    r: int,
    drift_node: int | None,
    toward_fleet: bool = False,
) -> np.ndarray:
    """Return one round's probe scores; drift ramps over ten rounds from 15.

    By default the node drifts AWAY from the fleet (toward one class, like a
    biased sensor). With `toward_fleet` it drifts toward the uniform
    distribution, which is where the consensus of heterogeneous nodes sits.
    """
    out = base + rng.normal(0, 0.01, base.shape)
    if drift_node is not None and r >= 15:
        a = min(1.0, (r - 15 + 1) / 10) * 0.6
        target = np.full(C, 1.0 / C) if toward_fleet else np.eye(C)[0]
        out[drift_node] = (1 - a) * out[drift_node] + a * target
    out = np.clip(out, 1e-4, None)
    return out / out.sum(-1, keepdims=True)


def drive(
    s: IntegrityFedAvg,
    drift_node: int | None,
    seed: int = 0,
    toward_fleet: bool = False,
) -> None:
    """Feed the strategy R rounds of fit results, optionally with drift."""
    rng = np.random.default_rng(seed)
    base = healthy_probs(np.random.default_rng(99))
    for r in range(R):
        probs = round_probs(base, rng, r, drift_node, toward_fleet)
        results = [fitres(k, probs[k], float(k)) for k in range(K)]
        s.aggregate_fit(r + 1, results, [])


def test_payload_roundtrip_and_size():
    """Probe payload is float16 on the wire and decodes to within 1e-3."""
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(C), size=P).astype(np.float32)
    blob = encode_scores(p)
    assert len(blob) == P * C * 2  # float16
    np.testing.assert_allclose(decode_scores(blob, P, C), p, atol=1e-3)


def test_fingerprint_mismatch_is_recorded_not_scored():
    """A report with the wrong battery fingerprint is logged, not scored."""
    s = strategy()
    p = np.full((P, C), 1.0 / C)
    s.aggregate_fit(1, [fitres(0, p, 0.0), fitres(1, p, 1.0, fp="WRONG")], [])
    assert s.state.fingerprint_mismatches == [(0, 1)]
    assert set(s.state.probs[0]) == {0}


def test_quarantine_excludes_flagged_node_from_the_average():
    """With quarantine on, a flagged node is left out of the average."""
    s = strategy(quarantine=True)
    s._ever_flagged = {4}
    p = np.full((P, C), 1.0 / C)
    results = [fitres(k, p, float(k)) for k in range(K)]
    params, _ = s.aggregate_fit(1, results, [])
    # Equal weights; nodes 0-3 average to 1.5, node 4 (value 4.0) excluded.
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], 1.5)
    assert s.state.quarantined[0] == [4]


def test_quarantine_off_keeps_every_node():
    """With quarantine off, a flagged node still enters the average."""
    s = strategy(quarantine=False)
    s._ever_flagged = {4}
    p = np.full((P, C), 1.0 / C)
    results = [fitres(k, p, float(k)) for k in range(K)]
    params, _ = s.aggregate_fit(1, results, [])
    np.testing.assert_allclose(parameters_to_ndarrays(params)[0], 2.0)


def test_detector_flags_the_drifting_node_and_no_healthy_node():
    """The first flag after onset names the drifting node and only it."""
    healthy = strategy()
    drive(healthy, None, seed=1)
    c = healthy.cusum_so_far()
    thr = float(np.nanmax(c[15:]))

    faulted = strategy(threshold=thr)
    drive(faulted, drift_node=2, seed=2)
    first = min(r for r, ks in faulted.state.flagged.items() if ks)
    assert first >= 15  # nothing before onset
    # The first flag is the faulted node, alone.
    assert faulted.state.flagged[first] == [2]
    # Later rounds may also flag healthy nodes: a far-drifted node moves the
    # leave-one-out consensus every peer is measured against. That is
    # CUSUM's documented mislocalization (README, stage 4); localization
    # claims rest on the FIRST flag, and quarantine acts on it.


def test_known_limit_drift_toward_the_fleet_is_not_flagged():
    """A node drifting toward the consensus is not flagged (known limit).

    The failure mode the brief discloses (gain drift on images): a node whose
    fault moves it TOWARD the consensus lowers its own divergence, and a
    one-sided CUSUM accumulates only increases. Pinned here so a future
    change that 'fixes' it has to do so deliberately, and so the claim stays
    honest.
    """
    healthy = strategy()
    drive(healthy, None, seed=1)
    thr = float(np.nanmax(healthy.cusum_so_far()[15:]))
    faulted = strategy(threshold=thr)
    drive(faulted, drift_node=2, seed=2, toward_fleet=True)
    assert not any(2 in ks for ks in faulted.state.flagged.values())
