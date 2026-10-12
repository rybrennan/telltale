from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.config import FedConfig
from telltale.detect import (
    LN2,
    characterise,
    check_reports,
    cusum,
    js,
    kl,
    level_divergence,
    level_series,
    loo_consensus,
    offset_change,
    persistence,
    persistence_series,
    robust_z,
    scorer_is_deterministic,
    self_divergence,
    self_referenced_level,
)
from telltale.fedavg import run_federated
from telltale.models import build_model
from telltale.probes import (
    ProbeConfig,
    ProbeMonitor,
    ProbeReport,
    build_probe_battery,
)

# ── divergences ───────────────────────────────────────────────────────────────


def test_kl_and_js_known_values():
    """KL and JS match hand values, are zero on self, and JS is symmetric."""
    p = np.array([[0.5, 0.5]])
    q = np.array([[0.9, 0.1]])
    expected_kl = 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1)
    assert kl(p, q)[0] == pytest.approx(expected_kl)
    assert kl(p, p)[0] == pytest.approx(0.0, abs=1e-12)
    assert js(p, p)[0] == pytest.approx(0.0, abs=1e-12)
    assert js(p, q)[0] == pytest.approx(js(q, p)[0])
    # Disjoint support saturates JS at ln 2.
    disjoint = js(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
    assert disjoint[0] == pytest.approx(LN2, rel=1e-6)


def test_js_is_rowwise():
    """JS returns one divergence per row."""
    p = np.array([[0.5, 0.5], [1.0, 0.0]])
    q = np.array([[0.5, 0.5], [0.0, 1.0]])
    out = js(p, q)
    assert out.shape == (2,)
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[1] == pytest.approx(LN2, rel=1e-6)


# ── consensus and level ───────────────────────────────────────────────────────


def test_loo_consensus_excludes_self():
    """Leave-one-out consensus averages every client except the one scored."""
    # 3 clients, 1 probe, 2 classes
    p = np.array([[[1.0, 0.0]], [[0.0, 1.0]], [[0.5, 0.5]]])
    q = loo_consensus(p)
    np.testing.assert_allclose(q[0], [[0.25, 0.75]])
    np.testing.assert_allclose(q[1], [[0.75, 0.25]])
    np.testing.assert_allclose(q[2], [[0.5, 0.5]])


def test_loo_consensus_handles_absent_client():
    """An absent (NaN) client is skipped in its peers' consensus."""
    p = np.array([[[1.0, 0.0]], [[np.nan, np.nan]], [[0.0, 1.0]]])
    q = loo_consensus(p)
    np.testing.assert_allclose(q[0], [[0.0, 1.0]])
    np.testing.assert_allclose(q[2], [[1.0, 0.0]])
    # Absent: plain mean of the present.
    np.testing.assert_allclose(q[1], [[0.5, 0.5]])


def test_level_divergence_isolates_the_odd_client():
    """A client whose rows differ from every peer's has the largest level.

    Numerical correctness of the estimator, not a detection claim.
    """
    rng = np.random.default_rng(0)
    base = rng.dirichlet(np.ones(4), size=20)  # 20 probes, 4 classes
    probs = np.stack(
        [base + rng.normal(0, 0.01, base.shape) for _ in range(6)]
    )
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum(axis=-1, keepdims=True)
    probs[3] = np.roll(probs[3], 1, axis=-1)  # client 3 believes something else
    level, offset = level_divergence(probs)
    assert np.argmax(level) == 3
    assert offset.shape == probs.shape
    np.testing.assert_allclose(offset[0], probs[0] - loo_consensus(probs)[0])


def test_robust_z_known_values_and_degenerate_cases():
    """Robust z uses median and MAD, and is 0 or NaN in degenerate cases."""
    z, med, scale = robust_z(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert med == 3.0
    assert scale == pytest.approx(1.4826 * 1.0)
    assert z[4] == pytest.approx(97.0 / 1.4826)
    assert z[2] == 0.0
    z, _, _ = robust_z(np.array([2.0, 2.0, 2.0]))
    assert np.all(z == 0.0)
    z, _, _ = robust_z(np.array([1.0, np.nan]))
    assert np.isnan(z).all()


def test_level_series_shapes_and_nan_pattern():
    """level_series returns the right shapes and NaN only for absent reports."""
    R, K, n, C = 3, 4, 5, 3
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(C), size=(R, K, n))
    probs[1, 2] = np.nan
    out = level_series(probs)
    assert out["level"].shape == (R, K)
    assert out["z"].shape == (R, K)
    assert out["offset"].shape == (R, K, n, C)
    assert np.isnan(out["level"][1, 2]) and np.isnan(out["z"][1, 2])
    assert np.isfinite(out["level"][0]).all()
    assert out["median"].shape == (R,)


# ── temporal self ─────────────────────────────────────────────────────────────


def test_self_divergence_zero_when_unchanged_positive_when_changed():
    """Self-divergence is 0 for an unchanged client and positive on change."""
    R, K, n, C = 3, 2, 4, 3
    rng = np.random.default_rng(2)
    probs = np.repeat(rng.dirichlet(np.ones(C), size=(1, K, n)), R, axis=0)
    probs[2, 1] = np.roll(probs[2, 1], 1, axis=-1)
    sd = self_divergence(probs)
    assert np.isnan(sd[0]).all()
    assert sd[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert sd[2, 0] == pytest.approx(0.0, abs=1e-12)
    assert sd[2, 1] > 0.0


# ── offset change and persistence ─────────────────────────────────────────────


def test_offset_change_subtracts_own_trailing_mean():
    """Offset change is the offset minus its own trailing-window mean."""
    R, K, n, C = 6, 1, 1, 2
    offset = np.zeros((R, K, n, C))
    offset[:, 0, 0, 0] = [1, 2, 3, 4, 5, 6]
    ch = offset_change(offset, window=3)
    assert np.isnan(ch[:3]).all()
    assert ch[3, 0, 0, 0] == pytest.approx(4 - 2.0)
    assert ch[5, 0, 0, 0] == pytest.approx(6 - 4.0)
    with pytest.raises(ValueError):
        offset_change(offset, window=0)


def test_persistence_separates_directional_shift_from_noise():
    """Persistence scores a directional shift above zero-mean noise.

    The core estimation problem, on synthetic vectors: the same total
    energy, one directional and one zero-mean.
    """
    rng = np.random.default_rng(3)
    T, d = 8, 50
    noise = rng.normal(size=(T, d))
    shift = np.tile(rng.normal(size=(1, d)), (T, 1))
    rho_noise = persistence(noise)
    rho_shift = persistence(shift)
    assert rho_shift == pytest.approx(1.0)
    assert rho_noise < 0.6  # expected ≈ 1/sqrt(8) ≈ 0.35
    assert rho_shift > rho_noise
    assert np.isnan(persistence(np.zeros((T, d))))
    assert np.isnan(persistence(np.full((T, d), np.nan)))


def test_persistence_series_windows_and_nans():
    """persistence_series windows correctly and is NaN where inputs are."""
    R, K, n, C = 6, 2, 3, 2
    rng = np.random.default_rng(4)
    change = rng.normal(size=(R, K, n, C))
    change[:2] = np.nan
    ps = persistence_series(change, span=3)
    assert ps.shape == (R, K)
    # Rounds 0-1 NaN, and windows touching them.
    assert np.isnan(ps[:4]).all()
    assert np.isfinite(ps[4:]).all()
    assert ps[5, 0] == pytest.approx(persistence(change[3:6, 0]))
    with pytest.raises(ValueError):
        persistence_series(change, span=1)


# ── instrument self-check ─────────────────────────────────────────────────────


def _clean_report(battery, cid=0, rnd=1):
    """Return a valid report of uniform probabilities for the battery."""
    C, n = battery.n_classes, battery.n_probes
    p = torch.full((n, C), 1.0 / C)
    return ProbeReport(cid, rnd, battery.fingerprint, p)


def test_check_reports_is_clean_on_a_good_report():
    """A valid report raises no instrument issues."""
    ds = tiny_dataset()
    battery = build_probe_battery(ds, ProbeConfig(n_probes=8))
    assert check_reports([_clean_report(battery)], battery) == []


def test_check_reports_flags_each_failure_mode():
    """Each bad-report kind is flagged; a fresh report after a replay is not."""
    ds = tiny_dataset()
    battery = build_probe_battery(ds, ProbeConfig(n_probes=8))
    C = battery.n_classes

    bad_fp = _clean_report(battery)
    bad_fp.battery_fingerprint = "deadbeef"
    assert any("fingerprint" in s for s in check_reports([bad_fp], battery))

    bad_shape = _clean_report(battery)
    bad_shape.probs = torch.full((7, C), 1.0 / C)
    assert any("shape" in s for s in check_reports([bad_shape], battery))

    nan = _clean_report(battery)
    nan.probs = nan.probs.clone()
    nan.probs[0, 0] = float("nan")
    assert any("non-finite" in s for s in check_reports([nan], battery))

    unnorm = _clean_report(battery)
    unnorm.probs = unnorm.probs * 1.5
    assert any("sum to 1" in s for s in check_reports([unnorm], battery))

    replay = _clean_report(battery)
    prev = {0: replay.probs.clone()}
    replay_issues = check_reports([replay], battery, previous=prev)
    assert any("replay" in s for s in replay_issues)
    fresh = _clean_report(battery)
    fresh.probs = fresh.probs.clone()
    fresh.probs[0] = torch.tensor([0.5, 0.5, 0.0, 0.0])
    assert check_reports([fresh], battery, previous=prev) == []


def test_scorer_is_deterministic_on_cpu():
    """Probe scoring gives identical output on repeat runs on CPU."""
    ds = tiny_dataset()
    battery = build_probe_battery(ds, ProbeConfig(n_probes=8))
    model = build_model("small_cnn", ds.n_classes, seed=0)
    assert scorer_is_deterministic(model, battery)


# ── end to end on the fixture ─────────────────────────────────────────────────


def test_healthy_fixture_run_yields_finite_statistics_and_clean_instrument():
    """Healthy fixture run gives finite statistics and no instrument issues.

    Verifies plumbing on the synthetic fixture only. The fixture is small
    enough to memorise, so nothing here is a statement about the healthy
    divergence distribution: that belongs to the FashionMNIST runs.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=6)
    battery = build_probe_battery(ds, ProbeConfig(n_probes=16))
    monitor = ProbeMonitor(battery)
    run_federated(cfg, ds, equal_parts(ds.n_train, 4), update_hook=monitor)

    prev: dict[int, torch.Tensor] = {}
    for _rnd, reps in sorted(monitor.by_round().items()):
        assert check_reports(reps, battery, previous=prev) == []
        prev = {r.client_id: r.probs for r in reps}

    probs = monitor.probs_array(cfg.rounds, cfg.n_clients)
    rows = characterise(
        probs, sizes=[4, 8, 16], tail_rounds=3, window=2, span=2
    )
    assert [r["n_probes"] for r in rows] == [4, 8, 16]
    for r in rows:
        for k, v in r.items():
            assert np.isfinite(v), (k, v)
        assert 0.0 <= r["level_median"] <= LN2
        assert 0.0 <= r["persist_max"] <= 1.0 + 1e-9


# ── self-referenced level ─────────────────────────────────────────────────────


def test_self_referenced_level_is_own_history_z():
    """Self-referenced level is a z-score against the node's own history."""
    level = np.zeros((8, 2))
    level[:, 0] = [1, 2, 3, 2, 1, 2, 9, 2]
    level[:, 1] = 5.0  # constant: zero trailing SD -> NaN, never a flag
    z = self_referenced_level(level, window=5)
    assert np.isnan(z[:5]).all()
    trailing = level[1:6, 0]
    assert z[6, 0] == pytest.approx(
        (9 - trailing.mean()) / trailing.std(ddof=1)
    )
    assert np.isnan(z[6, 1])
    with pytest.raises(ValueError):
        self_referenced_level(level, window=1)


# ── cusum ─────────────────────────────────────────────────────────────────────


def test_cusum_accumulates_sustained_shift_and_forgets_spikes():
    """CUSUM grows under a sustained shift and decays after a single spike."""
    rng = np.random.default_rng(0)
    R = 30
    level = np.zeros((R, 3))
    level[:, 0] = 1.0 + rng.normal(0, 0.1, R)  # stationary
    level[:, 1] = 1.0 + rng.normal(0, 0.1, R)
    level[15:, 1] += 0.1  # sustained +1 sigma from round 15
    level[:, 2] = 1.0 + rng.normal(0, 0.1, R)
    level[15, 2] += 0.5  # one-round spike only
    S = cusum(level, reference=slice(2, 12), k=0.5)
    assert np.isnan(S[:12]).all()
    assert np.nanmax(S[:, 0]) < 3.0  # healthy stays small
    # Ramp accumulates ~0.5/round.
    assert S[-1, 1] > S[-1, 0] and S[-1, 1] > 3.0
    assert S[16, 2] > 0.0 and S[-1, 2] < S[-1, 1]  # spike decays
    with pytest.raises(ValueError):
        cusum(level, reference=slice(2, 3))


def test_two_sided_cusum_sees_a_sustained_decrease():
    """Two-sided CUSUM catches a sustained drop the one-sided form misses."""
    rng = np.random.default_rng(5)
    R = 30
    level = 1.0 + rng.normal(0, 0.1, (R, 2))
    level[15:, 1] -= 0.15  # sustained drop of 1.5 sigma on node 1
    one = cusum(level, slice(2, 12))
    two = cusum(level, slice(2, 12), two_sided=True)
    assert np.nanmax(one[:, 1]) < 3.0  # one-sided cannot see it
    assert two[-1, 1] > 5.0  # two-sided accumulates it
    # The faulted node dominates the healthy one.
    assert two[-1, 1] > 1.5 * np.nanmax(two[:, 0])
