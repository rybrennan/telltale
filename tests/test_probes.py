from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.config import FedConfig
from telltale.fedavg import run_federated
from telltale.models import build_model
from telltale.probes import (
    ProbeConfig,
    ProbeMonitor,
    build_probe_battery,
    model_bytes,
    payload_bytes,
    quantise,
    score_probes,
)

# ── battery construction ──────────────────────────────────────────────────────


def test_battery_is_stratified_round_robin():
    """Battery cycles through classes and draws distinct test samples."""
    ds = tiny_dataset()
    b = build_probe_battery(ds, ProbeConfig(n_probes=16))
    assert b.n_probes == 16
    assert b.y.tolist() == [i % ds.n_classes for i in range(16)]
    assert np.unique(b.indices).size == 16
    torch.testing.assert_close(b.x, ds.test_x[b.indices])


def test_battery_is_deterministic_given_seed_and_differs_across_seeds():
    """Same probe seed, same battery; different seed, different battery."""
    ds = tiny_dataset()
    a = build_probe_battery(ds, ProbeConfig(n_probes=16, probe_seed=1))
    b = build_probe_battery(ds, ProbeConfig(n_probes=16, probe_seed=1))
    c = build_probe_battery(ds, ProbeConfig(n_probes=16, probe_seed=2))
    np.testing.assert_array_equal(a.indices, b.indices)
    assert a.fingerprint == b.fingerprint
    assert not np.array_equal(a.indices, c.indices)
    assert a.fingerprint != c.fingerprint


def test_smaller_battery_is_a_prefix_of_larger():
    """A smaller battery is a prefix of a larger one with the same seed.

    The property the bandwidth curve rests on: one run over the large
    battery yields the statistic at every smaller size.
    """
    ds = tiny_dataset()
    big = build_probe_battery(ds, ProbeConfig(n_probes=40))
    small = build_probe_battery(ds, ProbeConfig(n_probes=12))
    np.testing.assert_array_equal(big.indices[:12], small.indices)
    pre = big.prefix(12)
    np.testing.assert_array_equal(pre.indices, small.indices)
    assert pre.fingerprint == small.fingerprint
    assert pre.fingerprint != big.fingerprint


def test_battery_from_train_split_and_too_large_raises():
    """Battery can draw from train, and asking for too many probes raises."""
    ds = tiny_dataset()
    b = build_probe_battery(ds, ProbeConfig(n_probes=8, source="train"))
    torch.testing.assert_close(b.x, ds.train_x[b.indices])
    with pytest.raises(ValueError, match="cannot build"):
        build_probe_battery(ds, ProbeConfig(n_probes=81))  # test split has 80


def test_probe_config_validation_and_fingerprint():
    """ProbeConfig rejects bad values and its fingerprint tracks the seed."""
    with pytest.raises(ValueError, match="n_probes"):
        ProbeConfig(n_probes=0)
    with pytest.raises(ValueError, match="source"):
        ProbeConfig(source="dev")
    with pytest.raises(ValueError, match="scalar_dtype"):
        ProbeConfig(scalar_dtype="bfloat16")
    assert ProbeConfig().fingerprint() == ProbeConfig().fingerprint()
    assert (
        ProbeConfig().fingerprint() != ProbeConfig(probe_seed=1).fingerprint()
    )


# ── scoring ───────────────────────────────────────────────────────────────────


def test_score_rows_are_distributions_and_deterministic():
    """Probe scores are finite float32 distributions and repeatable."""
    ds = tiny_dataset()
    b = build_probe_battery(ds, ProbeConfig(n_probes=16))
    model = build_model("small_cnn", ds.n_classes, seed=0)
    p = score_probes(model, b)
    assert p.shape == (16, ds.n_classes)
    assert p.dtype == torch.float32
    assert torch.isfinite(p).all()
    torch.testing.assert_close(p.sum(dim=1), torch.ones(16))
    assert torch.equal(p, score_probes(model, b))


def test_scoring_a_prefix_battery_matches_slicing_only_approximately():
    """Scoring a prefix battery matches slicing to 1e-6, not bitwise.

    Documents an instrument fact rather than a design wish: a different
    batch composition changes CPU GEMM blocking and moves outputs by ~1e-8.
    Sub-battery statistics are therefore taken by slicing one run's reports,
    never by re-scoring, and bitwise checks only hold within a run.
    """
    ds = tiny_dataset()
    big = build_probe_battery(ds, ProbeConfig(n_probes=40))
    model = build_model("small_cnn", ds.n_classes, seed=0)
    full = score_probes(model, big)
    part = score_probes(model, big.prefix(12))
    torch.testing.assert_close(part, full[:12], rtol=0, atol=1e-6)


def test_quantise_rounds_to_wire_precision():
    """Quantise rounds scores to the wire dtype and rejects unknown ones."""
    p = np.array([[0.123456789, 0.876543211]], dtype=np.float32)
    assert np.array_equal(quantise(p, "float32"), p)
    f16 = quantise(p, "float16")
    assert f16.dtype == np.float32
    assert abs(f16[0, 0] - 0.123456789) < 1e-3 and f16[0, 0] != p[0, 0]
    u8 = quantise(p, "uint8")
    assert np.allclose(u8 * 255, np.round(u8 * 255))
    with pytest.raises(ValueError):
        quantise(p, "int4")


def test_payload_bytes_is_independent_of_model_and_data():
    """Payload size depends only on probes, classes and dtype."""
    ds = tiny_dataset()
    model = build_model("small_cnn", ds.n_classes, seed=0)
    assert payload_bytes(500, 10, "float16") == 10_000
    assert payload_bytes(50, 10, "uint8") == 500
    # SmallCNN with 4 classes: parameters only, all float32.
    n_params = sum(p.numel() for p in model.parameters())
    assert model_bytes(model.state_dict()) == 4 * n_params


# ── the monitor on the hook seam ──────────────────────────────────────────────


def test_monitor_does_not_alter_the_federation():
    """Attaching the monitor leaves the federation bitwise identical.

    The property everything downstream depends on. Asserted on parameters,
    not accuracy.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=16))

    plain = run_federated(cfg, ds, parts)
    monitor = ProbeMonitor(battery)
    watched = run_federated(cfg, ds, parts, update_hook=monitor)

    for k, v in plain.final_state.items():
        torch.testing.assert_close(v, watched.final_state[k], rtol=0, atol=0)
    assert plain.accuracy_curve() == watched.accuracy_curve()
    assert [r.mean_train_loss for r in plain.rounds] == [
        r.mean_train_loss for r in watched.rounds
    ]


def test_monitor_consumes_no_global_randomness():
    """Scoring a state does not advance the global torch RNG."""
    ds = tiny_dataset()
    battery = build_probe_battery(ds, ProbeConfig(n_probes=8))
    torch.manual_seed(7)
    before = torch.get_rng_state()
    monitor = ProbeMonitor(battery)
    state = build_model("small_cnn", ds.n_classes, seed=3).state_dict()
    torch.manual_seed(7)
    monitor.score_state(state)
    assert torch.equal(before, torch.get_rng_state())


def test_monitor_records_every_participant_every_round():
    """Monitor keeps one report per participant per round."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=12))
    monitor = ProbeMonitor(battery)
    run_federated(cfg, ds, equal_parts(ds.n_train, 4), update_hook=monitor)

    assert len(monitor.reports) == 12
    by_round = monitor.by_round()
    assert sorted(by_round) == [0, 1, 2]
    for reps in by_round.values():
        assert [r.client_id for r in reps] == [0, 1, 2, 3]
        for r in reps:
            assert r.battery_fingerprint == battery.fingerprint
            assert r.probs.shape == (12, ds.n_classes)

    arr = monitor.probs_array(n_rounds=3, n_clients=4)
    assert arr.shape == (3, 4, 12, ds.n_classes)
    assert np.isfinite(arr).all()
    # Reports are the node's *post-training* state: they change round to
    # round.
    assert not np.array_equal(arr[0, 0], arr[1, 0])


def test_monitor_leaves_nan_for_absent_clients():
    """Clients absent from a round are NaN in the probs array."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=2, client_fraction=0.5)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=8))
    monitor = ProbeMonitor(battery)
    run_federated(cfg, ds, equal_parts(ds.n_train, 4), update_hook=monitor)
    arr = monitor.probs_array(n_rounds=2, n_clients=4)
    present = ~np.isnan(arr).any(axis=(2, 3))
    assert present.sum(axis=1).tolist() == [2, 2]


def test_monitor_scores_the_returned_state_not_the_global():
    """Monitor scores each node's returned state, not the global model.

    The report must reflect what the node will contribute, i.e. its
    post-local-training state, so two nodes with different shards report
    differently in the same round.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=1)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=16))
    monitor = ProbeMonitor(battery)
    run_federated(cfg, ds, equal_parts(ds.n_train, 4), update_hook=monitor)
    arr = monitor.probs_array(1, 4)
    assert not np.array_equal(arr[0, 0], arr[0, 1])
