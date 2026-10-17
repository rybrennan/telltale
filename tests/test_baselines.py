from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.baselines import (
    UpdateStatsMonitor,
    fedrr_cusum,
    fedrr_threshold,
    fit_subspace,
    flatten,
    in_control_arl,
    local_steps,
    rank_scores,
    residual,
)
from telltale.config import FedConfig
from telltale.fedavg import run_federated
from telltale.models import build_model

# ── the monitor on a real (tiny) federation ───────────────────────────────────


def _run(cfg, ds, parts, hook=None):
    return run_federated(cfg, ds, parts, update_hook=hook)


def test_monitor_is_inert():
    """A run with the monitor attached is bitwise identical to one without."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=4, local_epochs=1, batch_size=32)
    parts = equal_parts(ds.n_train, 4)
    plain = _run(cfg, ds, parts)
    mon = UpdateStatsMonitor(4, 4, ds.n_classes, cfg.init_seed, range(1, 3))
    watched = _run(cfg, ds, parts, mon)
    assert plain.accuracy_curve() == watched.accuracy_curve()
    for k, v in plain.final_state.items():
        torch.testing.assert_close(v, watched.final_state[k], rtol=0, atol=0)


def test_monitor_tracks_the_true_global_state():
    """Round-0 deltas are taken from the run's real initial model.

    The monitor rebuilds the first global state from the init seed. If that
    reconstruction were wrong, round 0's norms would not match updates
    measured against the model `run_federated` actually broadcast.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=3, rounds=2, local_epochs=1, batch_size=32)
    parts = equal_parts(ds.n_train, 3)
    seen = {}

    def grab(updates, rnd):
        seen[rnd] = [flatten(u.state) for u in updates]

    mon = UpdateStatsMonitor(2, 3, ds.n_classes, cfg.init_seed, range(1, 2))

    def both(updates, rnd):
        grab(updates, rnd)
        mon(updates, rnd)

    _run(cfg, ds, parts, both)
    init = flatten(
        build_model("small_cnn", ds.n_classes, cfg.init_seed).state_dict()
    )
    for k in range(3):
        expect = float(torch.linalg.vector_norm(seen[0][k] - init))
        assert mon.norm[0, k] == pytest.approx(expect, rel=1e-12)


def test_monitor_fills_residuals_only_after_the_fit():
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=5, local_epochs=1, batch_size=32)
    mon = UpdateStatsMonitor(5, 4, ds.n_classes, cfg.init_seed, range(1, 3))
    _run(cfg, ds, equal_parts(ds.n_train, 4), mon)
    a = mon.arrays()
    assert np.isnan(a["resid"][:3]).all()
    assert np.isfinite(a["resid"][3:]).all()
    assert np.isfinite(a["norm"]).all() and np.isfinite(a["loo_dist"]).all()
    assert 1 <= int(a["q"]) <= 8  # at most rounds-in-fit x clients
    assert 0.95 <= float(a["held"]) <= 1.0


# ── subspace and residual ─────────────────────────────────────────────────────


def test_residual_is_zero_inside_the_subspace_and_full_outside():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(3, 50, generator=g, dtype=torch.float64)
    basis, q, held = fit_subspace(a, energy=1.0)
    assert q == 3 and held == pytest.approx(1.0)
    inside = 2.0 * a[0] - a[2]
    assert residual(inside, basis) == pytest.approx(0.0, abs=1e-9)
    outside = torch.randn(50, generator=g, dtype=torch.float64)
    outside -= basis @ (basis.T @ outside)
    assert residual(outside, basis) == pytest.approx(
        float(torch.linalg.vector_norm(outside)), rel=1e-9
    )


def test_fit_subspace_keeps_the_fewest_directions_for_the_energy():
    # Two strong directions and one weak one: 95% needs two.
    e = torch.eye(3, 10, dtype=torch.float64)
    deltas = torch.stack([10 * e[0], 10 * e[1], 1 * e[2]])
    _, q, held = fit_subspace(deltas, energy=0.95)
    assert q == 2
    assert held == pytest.approx(200 / 201)


def test_fit_subspace_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_subspace(torch.zeros(0, 5))
    with pytest.raises(ValueError):
        fit_subspace(torch.ones(2, 5), energy=0.0)


# ── FedRR's statistic ────────────────────────────────────────────────────────


def test_rank_scores_are_ordered_and_skip_absent_clients():
    rng = np.random.default_rng(0)
    z = rank_scores(np.array([0.3, np.nan, 0.9, 0.1]), rng)
    assert np.isnan(z[1])
    assert z[2] > z[0] > z[3]


def test_rank_scores_are_standard_normal_under_random_ranks():
    rng = np.random.default_rng(1)
    z = np.concatenate(
        [
            rank_scores(rng.permutation(10).astype(float), rng)
            for _ in range(4000)
        ]
    )
    assert abs(z.mean()) < 0.03
    assert abs(z.std() - 1.0) < 0.03


def test_fedrr_cusum_grows_on_a_client_always_ranked_highest():
    R, K = 20, 5
    sig = np.tile(np.arange(K, dtype=float), (R, 1))  # client 4 always largest
    s = fedrr_cusum(sig, start=5, d=0.5)
    assert np.isnan(s[:5]).all()
    assert (np.diff(s[5:, 4]) > 0).all()
    assert s[-1, 0] == 0.0  # always ranked lowest: never accumulates


def test_fedrr_cusum_is_reproducible_from_its_seed():
    sig = np.random.default_rng(3).normal(size=(12, 6))
    np.testing.assert_array_equal(
        fedrr_cusum(sig, 2, seed=7), fedrr_cusum(sig, 2, seed=7)
    )


def test_fedrr_threshold_hits_its_target_run_length():
    h = fedrr_threshold(30, n_clients=10, d=0.5, runs=400, seed=0)
    arl = in_control_arl(h, 10, 0.5, 2000, np.random.default_rng(99))
    assert 24 <= arl <= 36  # fresh simulation, within sampling error
    assert fedrr_threshold(60, 10, 0.5, runs=400, seed=0) > h


def test_local_steps_counts_batches_per_epoch():
    np.testing.assert_array_equal(
        local_steps([64, 65, 1149], batch_size=64, local_epochs=1), [1, 2, 18]
    )
    np.testing.assert_array_equal(
        local_steps([100], batch_size=64, local_epochs=3), [6]
    )


def test_step_normalisation_removes_the_size_ranking():
    """A client whose update is long only because it took more steps.

    Raw sizes rank the big client first every round; per-step sizes are
    identical, so after normalisation it holds no fixed rank.
    """
    steps = local_steps([640, 64, 64, 64], batch_size=64, local_epochs=1)
    raw = np.tile(steps * 0.1, (30, 1))  # same per-step size for everyone
    raw += np.random.default_rng(0).normal(scale=1e-3, size=raw.shape)
    s_raw = fedrr_cusum(raw, start=0)
    s_norm = fedrr_cusum(raw / steps, start=0)
    assert s_raw[-1, 0] > 10  # always ranked first: grows every round
    assert s_norm[-1, 0] < s_raw[-1, 0] / 3


def test_exported_probe_statistics_score_identically(tmp_path):
    """Scoring from exported statistics equals scoring from the array."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from score_faults import probe_statistics

    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=(20, 5, 8)).astype(np.float32)
    classes = np.array(["ship"] * 5)
    direct = probe_statistics(probs, classes)
    np.savez_compressed(tmp_path / "s.stats.npz", **direct)
    with np.load(tmp_path / "s.stats.npz") as z:
        loaded = probe_statistics({k: z[k] for k in z.files}, classes)
    assert direct.keys() == loaded.keys()
    for k in direct:
        np.testing.assert_array_equal(direct[k], loaded[k])


def test_grid_plan_filters_by_alpha():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from run_grid import plan

    every = plan(seeds=[0])
    some = plan(seeds=[0], alphas=[0.5, 100.0])
    assert len(every) == 33 and len(some) == 22
    assert all(
        "--alpha" in a and a[a.index("--alpha") + 1] != "0.1" for _, a in some
    )
    nodes = plan(extra_nodes=True, seeds=[0, 1, 2], alphas=[0.5])
    assert len(nodes) == 3 * (1 + 10 + 8)


def test_load_prefers_exported_statistics_over_a_raw_array(
    tmp_path, monkeypatch
):
    """A leftover raw array must not override the published statistics."""
    import json as _json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import score_faults as sf

    monkeypatch.setattr(sf, "RESULTS", tmp_path)
    (tmp_path / "r.json").write_text(_json.dumps({"config": {}}))
    np.savez_compressed(tmp_path / "r.npz", probs=np.zeros((2, 2, 1, 2)))
    np.savez_compressed(tmp_path / "r.stats.npz", level=np.ones((2, 2)))
    _, probs, _ = sf.load("r")
    assert isinstance(probs, dict) and probs["level"].sum() == 4


def test_false_alarm_rate_ignores_rounds_where_the_statistic_is_undefined():
    """A CUSUM cannot fire before its reference epoch ends; those rounds
    must not dilute its false-alarm rate."""
    from telltale.metrics import (
        Threshold,
        false_alarm_rate,
        node_round_rate,
        round_alarm_rate,
    )

    series = np.full((20, 2), np.nan)
    series[15:] = 0.0
    series[16, 0] = 5.0  # one alarm on one node
    thr = Threshold("cusum", 1.0, 10, "max")
    flagged = np.zeros_like(series, dtype=bool)
    flagged[16, 0] = True
    # 5 live rounds x 2 nodes = 10 defined node-rounds; 1 flagged
    assert node_round_rate(flagged, series, 10) == 0.1
    assert false_alarm_rate(series, thr) == 0.1
    # 5 live rounds, 1 with an alarm
    assert round_alarm_rate(flagged, series, 10) == 0.2


# ── review 4: fingerprints, clipping, FedRR allowance ─────────────────────────


def test_existing_fingerprints_are_unchanged_by_the_clip_field():
    """Result files are named by fingerprint; adding clip_norm must not
    rename any of them."""
    assert FedConfig(dirichlet_alpha=0.5, tag="baseline").fingerprint() == (
        "dbececbce9b8"
    )
    assert FedConfig(dirichlet_alpha=0.1, tag="severe").fingerprint() == (
        "6b496a47c12d"
    )
    assert FedConfig(dirichlet_alpha=100.0, tag="iid").fingerprint() == (
        "cc591762a646"
    )
    assert FedConfig(dirichlet_alpha=0.5, clip_norm=1.0).fingerprint() != (
        FedConfig(dirichlet_alpha=0.5).fingerprint()
    )


def test_clipping_bounds_every_sent_update():
    ds = tiny_dataset()
    cfg = FedConfig(
        n_clients=4, rounds=3, local_epochs=1, batch_size=32, clip_norm=0.01
    )
    mon = UpdateStatsMonitor(3, 4, ds.n_classes, cfg.init_seed, range(1, 2))
    run_federated(cfg, ds, equal_parts(ds.n_train, 4), update_hook=mon)
    assert np.nanmax(mon.norm) <= 0.01 * (1 + 1e-6)


def test_a_loose_clip_is_bitwise_identical_to_no_clip():
    ds = tiny_dataset()
    parts = equal_parts(ds.n_train, 4)
    base = FedConfig(n_clients=4, rounds=3, local_epochs=1, batch_size=32)
    loose = FedConfig(
        n_clients=4, rounds=3, local_epochs=1, batch_size=32, clip_norm=1e9
    )
    a = run_federated(base, ds, parts)
    b = run_federated(loose, ds, parts)
    assert a.accuracy_curve() == b.accuracy_curve()
    for k, v in a.final_state.items():
        torch.testing.assert_close(v, b.final_state[k], rtol=0, atol=0)


def test_clipping_leaves_the_probe_scoring_model_unclipped():
    """Probes score the client's local model; only what it sends is clipped."""
    ds = tiny_dataset()
    cfg = FedConfig(
        n_clients=3, rounds=1, local_epochs=1, batch_size=32, clip_norm=1e-6
    )
    seen = {}

    def grab(updates, rnd):
        seen["local"] = flatten(updates[0].state)
        seen["sent"] = flatten(updates[0].sent)

    run_federated(cfg, ds, equal_parts(ds.n_train, 3), update_hook=grab)
    assert not torch.equal(seen["local"], seen["sent"])


def test_fedrr_allowance_matches_the_authors_control_limits():
    """FedRR's published limits (K = 5, ARL0 = 30) reproduce only with the
    full d subtracted, which is what fedrr_cusum and in_control_arl do."""
    for h, d in ((3.84, 0.4), (3.28, 0.5), (2.77, 0.6)):
        arl = in_control_arl(h, 5, d, 1500, np.random.default_rng(1))
        assert 24 <= arl <= 38, (h, d, arl)


def test_review4_grid_cells_get_new_files_and_old_cells_keep_theirs():
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo / "scripts"))
    from run_grid import plan

    old = plan(seeds=[0], alphas=[0.5])
    assert all(
        (repo / "results" / f"{stem}.json").exists() for stem, _ in old
    ), "default grid stems must still point at the published result files"
    mid = plan(seeds=[0], alphas=[0.5], mid_only=True)
    assert len(mid) == 1 + 4
    for cell in (
        plan(seeds=[0], alphas=[0.5], mid_only=True, partition_seed=1),
        plan(seeds=[0], alphas=[0.5], mid_only=True, clip_norm=0.8),
    ):
        assert not {s for s, _ in cell} & {s for s, _ in old}
    late = plan(seeds=[0], alphas=[0.5], mid_only=True, onset=22)
    assert late[0][0] == mid[0][0]  # the healthy run is shared
    assert not {s for s, _ in late[1:]} & {s for s, _ in mid[1:]}
    clip = plan(seeds=[0], alphas=[0.5], mid_only=True, clip_norm=0.8)
    assert all("--clip-norm" in a for _, a in clip)
