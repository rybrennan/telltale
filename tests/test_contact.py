from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.config import FedConfig
from telltale.contact import ContactSchedule
from telltale.detect import (
    cusum,
    level_series,
    offset_change,
    persistence_series,
    self_divergence,
    self_referenced_level,
)
from telltale.faults import FaultSpec, make_fault
from telltale.fedavg import run_federated
from telltale.metrics import Threshold, score_fault_run
from telltale.probes import ProbeConfig, ProbeMonitor, build_probe_battery

# ── schedule ──────────────────────────────────────────────────────────────────


def test_schedule_staggers_submarines_and_keeps_ships_every_round():
    """Subs report on staggered phases; ships report every round."""
    sch = ContactSchedule(period=3, sub_nodes=(1, 3))
    got = [sch.participants(r, 4) for r in range(6)]
    # node 1 phase 0: rounds 0,3; node 3 phase 1: rounds 2,5
    assert got == [
        [0, 1, 2], [0, 2], [0, 2, 3], [0, 1, 2], [0, 2], [0, 2, 3]
    ]
    mask = sch.observed(6, 4)
    assert mask[:, 0].all() and mask[:, 2].all()
    assert mask[:, 1].tolist() == [True, False, False, True, False, False]
    other = ContactSchedule(period=2, sub_nodes=(1, 3))
    assert sch.fingerprint() != other.fingerprint()
    with pytest.raises(ValueError):
        ContactSchedule(period=0)
    with pytest.raises(ValueError):
        ContactSchedule(period=2, sub_nodes=(1, 1))


def test_all_present_schedule_is_bitwise_identical_to_none():
    """An everyone-present schedule matches no schedule bit for bit."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    plain = run_federated(cfg, ds, parts)
    sch = ContactSchedule(period=1, sub_nodes=(0, 1, 2, 3))
    scheduled = run_federated(
        cfg, ds, parts, participation=lambda r: sch.participants(r, 4)
    )
    for k, v in plain.final_state.items():
        torch.testing.assert_close(v, scheduled.final_state[k], rtol=0, atol=0)
    assert [r.participants for r in scheduled.rounds] == [[0, 1, 2, 3]] * 3


def test_silent_rounds_produce_no_report_and_change_the_federation():
    """Silent rounds leave NaN reports and change the trained model."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=4)
    parts = equal_parts(ds.n_train, 4)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=12))
    sch = ContactSchedule(period=2, sub_nodes=(2,))
    mon = ProbeMonitor(battery)
    res = run_federated(
        cfg,
        ds,
        parts,
        update_hook=mon,
        participation=lambda r: sch.participants(r, 4),
    )
    assert [r.participants for r in res.rounds] == [
        [0, 1, 2, 3], [0, 1, 3], [0, 1, 2, 3], [0, 1, 3]
    ]
    arr = mon.probs_array(4, 4)
    present = ~np.isnan(arr).any(axis=(2, 3))
    np.testing.assert_array_equal(present, sch.observed(4, 4))
    plain = run_federated(cfg, ds, parts)
    assert any(
        not torch.equal(plain.final_state[k], res.final_state[k])
        for k in plain.final_state
    )


def test_participation_validation():
    """Empty or out-of-range participant lists raise ValueError."""
    ds = tiny_dataset()
    with pytest.raises(ValueError, match="participation returned"):
        run_federated(
            FedConfig(n_clients=4, rounds=1),
            ds,
            equal_parts(ds.n_train, 4),
            participation=lambda r: [],
        )
    with pytest.raises(ValueError, match="participation returned"):
        run_federated(
            FedConfig(n_clients=4, rounds=1),
            ds,
            equal_parts(ds.n_train, 4),
            participation=lambda r: [7],
        )


# ── gap-tolerant statistics ───────────────────────────────────────────────────


def gapped(seed=0, R=14, K=3, n=4, C=3):
    """Return a dense series and a copy with node 1 silent on odd rounds.

    Every temporal statistic on the gapped series must equal the dense
    statistic computed over node 1's observed rounds only.
    """
    rng = np.random.default_rng(seed)
    dense = rng.dirichlet(np.ones(C), size=(R, K, n))
    sparse = dense.copy()
    sparse[1::2, 1] = np.nan
    return dense, sparse


def test_self_divergence_compares_to_previous_contact():
    """Self-divergence of a gapped node compares against its last contact."""
    dense, sparse = gapped()
    sd = self_divergence(sparse)
    # Node 1's observed rounds are the even ones.
    compact = self_divergence(dense[::2])
    np.testing.assert_allclose(sd[::2, 1][1:], compact[1:, 1])
    assert np.isnan(sd[1::2, 1]).all()
    np.testing.assert_allclose(sd[:, 0], self_divergence(dense)[:, 0])


def test_offset_change_and_persistence_run_over_contacts():
    """Offset change and persistence step over contacts, not raw rounds."""
    _, sparse = gapped()
    off_s = level_series(sparse)["offset"]
    ch = offset_change(off_s, window=2)
    ps = persistence_series(ch, span=2)
    # Node 1: same computation on its own observed offsets.
    obs = np.arange(0, 14, 2)
    own = off_s[obs, 1]
    expected = own[2] - own[:2].mean(axis=0)
    np.testing.assert_allclose(ch[obs[2], 1], expected)
    assert np.isnan(ch[1::2, 1]).all() and np.isnan(ps[1::2, 1]).all()
    assert np.isfinite(ps[obs[3:], 1]).all()


def test_self_referenced_level_and_cusum_skip_silent_rounds():
    """Self-referenced level and CUSUM are NaN on silent rounds only."""
    ramp = np.tile(np.linspace(1.0, 2.0, 12)[:, None], (1, 2))
    level = ramp + np.array([[0.0, 0.05]])
    level[1::2, 1] = np.nan
    z = self_referenced_level(level, window=3)
    S = cusum(level, reference=slice(0, 6), k=0.5)
    # Node 1's observed rounds are 0,2,4,...: statistics defined only there.
    assert np.isnan(z[1::2, 1]).all() and np.isnan(S[1::2, 1]).all()
    obs = np.arange(0, 12, 2)
    own = level[obs, 1]
    trailing = own[:3]
    np.testing.assert_allclose(
        z[obs[3], 1], (own[3] - trailing.mean()) / trailing.std(ddof=1)
    )
    # CUSUM reference for node 1 uses its contacts inside rounds 0-5:
    # rounds 0, 2, 4.
    mu, sd = own[:3].mean(), own[:3].std(ddof=1)
    expected = max(0.0, (own[3] - mu) / sd - 0.5)
    np.testing.assert_allclose(S[obs[3], 1], expected)
    # Dense node unchanged by the gaps in the other one.
    np.testing.assert_allclose(S[:, 0], cusum(level[:, :1], slice(0, 6))[:, 0])
    with pytest.raises(ValueError):
        cusum(level, reference=slice(3, 3))


# ── contacts to detection ─────────────────────────────────────────────────────


def test_contacts_to_detection_counts_target_reports():
    """Contacts-to-detection counts only rounds the target actually reported."""
    R, K, target, onset = 20, 3, 1, 10
    thr = Threshold("x", 0.5, burn_in=5, method="max")
    s = np.full((R, K), 0.1)
    s[16, target] = 0.9
    dense = score_fault_run(s, thr, target, onset)
    assert dense.time_to_detection == 6 and dense.contacts_to_detection == 7
    observed = ContactSchedule(period=3, sub_nodes=(target,)).observed(R, K)
    sparse = score_fault_run(s, thr, target, onset, observed=observed)
    assert sparse.contacts_to_detection == int(observed[onset:17, target].sum())
    assert sparse.contacts_to_detection < dense.contacts_to_detection


# ── end to end: fault on a submarine ──────────────────────────────────────────


def test_fault_on_a_submarine_only_touches_its_contact_rounds():
    """A fault on a sub shows up only from its first contact after onset."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=4)
    parts = equal_parts(ds.n_train, 4)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=12))
    sch = ContactSchedule(period=2, sub_nodes=(2,))
    spec = FaultSpec("bias", node=2, onset=1, severity=1.0)
    m_h, m_f = ProbeMonitor(battery), ProbeMonitor(battery)
    run_federated(
        cfg,
        ds,
        parts,
        update_hook=m_h,
        participation=lambda r: sch.participants(r, 4),
    )
    run_federated(
        cfg,
        ds,
        parts,
        update_hook=m_f,
        participation=lambda r: sch.participants(r, 4),
        fault=make_fault([spec], ds.n_classes),
    )
    h, f = m_h.probs_array(4, 4), m_f.probs_array(4, 4)
    np.testing.assert_array_equal(h[0], f[0])  # before onset
    # Onset round: sub silent, nothing differs.
    np.testing.assert_array_equal(h[1], f[1])
    assert np.isnan(f[1, 2]).all()
    # First contact after onset carries the fault.
    assert not np.array_equal(h[2, 2], f[2, 2])


def test_cusum_pooled_reference_sd_rescues_a_three_contact_boat():
    """Pooled reference SD stops a near-constant reference blowing up."""
    rng = np.random.default_rng(1)
    R, K = 30, 4
    level = 1.0 + rng.normal(0, 0.1, (R, K))
    # Nodes 2, 3 are boats on a 3-round schedule; node 3's three reference
    # points happen to be nearly identical, so its own SD is tiny.
    sch = ContactSchedule(period=3, sub_nodes=(2, 3))
    level[~sch.observed(R, K)] = np.nan
    ref3 = [r for r in range(5, 15) if sch.in_contact(3, r)]
    # Near-identical reference points.
    level[ref3, 3] = 1.0 + 0.001 * np.linspace(-1, 1, len(ref3))
    own = cusum(level, slice(5, 15))
    pools = np.array(["ship", "ship", "sub", "sub"])
    pooled = cusum(level, slice(5, 15), pool=pools)
    assert np.nanmax(own[:, 3]) > 10 * np.nanmax(pooled[:, 3])
    # Ships are untouched by pooling among themselves when their own SDs are
    # alike, and never touched by the boats' pool at all.
    ships_only = cusum(level[:, :2], slice(5, 15), pool=np.array(["s", "s"]))
    np.testing.assert_allclose(
        np.nanmax(pooled[:, :2], axis=0),
        np.nanmax(ships_only[:, :2], axis=0),
    )
    assert np.isnan(pooled[:, 3][~sch.observed(R, K)[:, 3]]).all()


def test_pooling_leaves_a_well_sampled_class_untouched():
    """Pooling a densely sampled class gives the same CUSUM as no pooling."""
    rng = np.random.default_rng(2)
    level = 1.0 + rng.normal(0, 0.1, (30, 4))
    dense = cusum(level, slice(5, 15))
    pooled = cusum(level, slice(5, 15), pool=np.array(["ship"] * 4))
    np.testing.assert_array_equal(np.nan_to_num(dense), np.nan_to_num(pooled))
