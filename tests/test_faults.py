from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.config import FedConfig
from telltale.faults import (
    FaultSpec,
    corrupt_labels,
    make_fault,
    transform_inputs,
)
from telltale.fedavg import run_federated
from telltale.probes import ProbeConfig, ProbeMonitor, build_probe_battery

# ── spec ──────────────────────────────────────────────────────────────────────


def test_spec_validation_and_fingerprint():
    """FaultSpec rejects bad fields and its fingerprint tracks every field."""
    with pytest.raises(ValueError, match="kind"):
        FaultSpec("fog", 0, 0, 1.0)
    with pytest.raises(ValueError, match="fraction"):
        FaultSpec("label_noise", 0, 0, 1.5)
    with pytest.raises(ValueError, match="ramp"):
        FaultSpec("bias", 0, 0, 1.0, ramp=-1)
    a = FaultSpec("bias", 2, 10, 0.5, ramp=5)
    same = FaultSpec("bias", 2, 10, 0.5, ramp=5)
    longer_ramp = FaultSpec("bias", 2, 10, 0.5, ramp=6)
    assert a.fingerprint() == same.fingerprint()
    assert a.fingerprint() != longer_ramp.fingerprint()


def test_magnitude_schedule_step_and_ramp():
    """Fault magnitude steps at onset, or ramps linearly over ramp rounds."""
    step = FaultSpec("bias", 0, onset=3, severity=2.0)
    assert [step.magnitude(r) for r in range(6)] == [0, 0, 0, 2.0, 2.0, 2.0]
    ramp = FaultSpec("bias", 0, onset=3, severity=2.0, ramp=4)
    got = [ramp.magnitude(r) for r in range(9)]
    assert got == pytest.approx([0, 0, 0, 0.5, 1.0, 1.5, 2.0, 2.0, 2.0])


# ── transforms ────────────────────────────────────────────────────────────────


def test_input_transforms_identity_at_zero_and_change_otherwise():
    """Input transforms are identity at zero and change the input otherwise."""
    x = torch.randn(6, 1, 28, 28)
    for kind in ("bias", "gain", "blur"):
        assert transform_inputs(x, kind, 0.0) is x
        out = transform_inputs(x, kind, 0.5)
        assert out.shape == x.shape
        assert not torch.equal(out, x)
        assert torch.isfinite(out).all()
    torch.testing.assert_close(transform_inputs(x, "bias", 0.25), x + 0.25)
    torch.testing.assert_close(transform_inputs(x, "gain", 0.25), x * 1.25)
    # Blur preserves the mean (kernel sums to 1) and reduces variance.
    b = transform_inputs(x, "blur", 1.0)
    assert b.mean().item() == pytest.approx(x.mean().item(), abs=2e-2)
    assert b.var() < x.var()
    with pytest.raises(ValueError):
        transform_inputs(x, "label_noise", 0.5)


def test_transform_does_not_mutate_input():
    """Input transforms leave the caller's tensor untouched."""
    x = torch.randn(4, 1, 28, 28)
    before = x.clone()
    transform_inputs(x, "bias", 1.0)
    transform_inputs(x, "blur", 1.0)
    assert torch.equal(x, before)


def test_label_noise_count_nesting_and_determinism():
    """Label noise flips the right count, nests by fraction, and is seeded."""
    y = torch.arange(10).repeat(20)  # 200 labels, 10 classes
    assert corrupt_labels(y, 0.0, 10, seed=0) is y
    a = corrupt_labels(y, 0.10, 10, seed=0)
    b = corrupt_labels(y, 0.25, 10, seed=0)
    changed_a = (a != y).nonzero().flatten()
    changed_b = (b != y).nonzero().flatten()
    assert changed_a.numel() == 20
    assert changed_b.numel() == 50
    assert set(changed_a.tolist()) <= set(changed_b.tolist())  # nested
    assert torch.equal(a, corrupt_labels(y, 0.10, 10, seed=0))
    assert not torch.equal(a, corrupt_labels(y, 0.10, 10, seed=1))
    assert ((a >= 0) & (a < 10)).all()
    assert torch.equal(y, torch.arange(10).repeat(20))  # input untouched


# ── the seam ──────────────────────────────────────────────────────────────────


def test_inactive_fault_is_bitwise_identical_to_no_fault():
    """A fault not yet started, or of zero severity, is the healthy run.

    The load-bearing property for stage 4: a run with a fault that has
    not started, or has severity zero, IS the healthy run.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    healthy = run_federated(cfg, ds, parts)

    not_yet = make_fault(
        [FaultSpec("bias", 1, onset=10, severity=1.0)], ds.n_classes
    )
    zero = make_fault(
        [FaultSpec("label_noise", 1, onset=0, severity=0.0)], ds.n_classes
    )
    for f in (not_yet, zero):
        r = run_federated(cfg, ds, parts, fault=f)
        for k, v in healthy.final_state.items():
            torch.testing.assert_close(v, r.final_state[k], rtol=0, atol=0)
        assert [x.mean_train_loss for x in r.rounds] == [
            x.mean_train_loss for x in healthy.rounds
        ]


def test_fault_touches_only_the_target_node_until_aggregation():
    """Only the target node differs from healthy until aggregation.

    In the onset round every node starts from the same global model, so
    non-target nodes must report bitwise-identical probe rows to the healthy
    run, and the target node must not.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=16))

    m_h = ProbeMonitor(battery)
    run_federated(cfg, ds, parts, update_hook=m_h)
    m_f = ProbeMonitor(battery)
    spec = FaultSpec("bias", node=2, onset=1, severity=1.0)
    run_federated(
        cfg,
        ds,
        parts,
        update_hook=m_f,
        fault=make_fault([spec], ds.n_classes),
    )

    h = m_h.probs_array(3, 4)
    f = m_f.probs_array(3, 4)
    # Round before onset: everyone identical.
    np.testing.assert_array_equal(h[0], f[0])
    for cid in (0, 1, 3):
        np.testing.assert_array_equal(h[1, cid], f[1, cid])
    assert not np.array_equal(h[1, 2], f[1, 2])
    # After aggregation the corrupted contribution has entered the global
    # model.
    assert not np.array_equal(h[2, 0], f[2, 0])


def test_faults_on_different_nodes_compose():
    """Faults on two nodes each hit their own node and leave the rest alone."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=2)
    parts = equal_parts(ds.n_train, 4)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=16))
    specs = [
        FaultSpec("gain", node=0, onset=0, severity=0.5),
        FaultSpec("label_noise", node=3, onset=0, severity=0.5),
    ]
    m_h, m_f = ProbeMonitor(battery), ProbeMonitor(battery)
    run_federated(cfg, ds, parts, update_hook=m_h)
    run_federated(
        cfg,
        ds,
        parts,
        update_hook=m_f,
        fault=make_fault(specs, ds.n_classes),
    )
    h, f = m_h.probs_array(2, 4), m_f.probs_array(2, 4)
    assert not np.array_equal(h[0, 0], f[0, 0])
    assert not np.array_equal(h[0, 3], f[0, 3])
    np.testing.assert_array_equal(h[0, 1], f[0, 1])
    np.testing.assert_array_equal(h[0, 2], f[0, 2])


def test_fault_run_is_reproducible():
    """A seeded faulted run reproduces bit for bit."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    spec = FaultSpec(
        "label_noise", node=1, onset=1, severity=0.3, ramp=2, seed=5
    )
    a = run_federated(cfg, ds, parts, fault=make_fault([spec], ds.n_classes))
    b = run_federated(cfg, ds, parts, fault=make_fault([spec], ds.n_classes))
    for k, v in a.final_state.items():
        torch.testing.assert_close(v, b.final_state[k], rtol=0, atol=0)
