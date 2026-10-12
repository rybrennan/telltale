from __future__ import annotations

import pytest
import torch
from helpers import equal_parts, tiny_dataset

from telltale.config import FedConfig
from telltale.fedavg import ClientUpdate, aggregate, evaluate, run_federated
from telltale.models import build_model

# ── aggregation ───────────────────────────────────────────────────────────────


def test_aggregate_equal_weights_is_plain_mean():
    """Equal sample counts aggregate to the plain mean."""
    a = {"w": torch.tensor([0.0, 10.0])}
    b = {"w": torch.tensor([2.0, 20.0])}
    out = aggregate(
        [
            ClientUpdate(0, 0, 50, 0.0, a),
            ClientUpdate(1, 0, 50, 0.0, b),
        ]
    )
    torch.testing.assert_close(out["w"], torch.tensor([1.0, 15.0]))


def test_aggregate_is_weighted_by_sample_count():
    """A client holding 3x the data pulls the average 3x as hard."""
    a = {"w": torch.tensor([0.0])}
    b = {"w": torch.tensor([4.0])}
    out = aggregate(
        [
            ClientUpdate(0, 0, 300, 0.0, a),
            ClientUpdate(1, 0, 100, 0.0, b),
        ]
    )
    torch.testing.assert_close(out["w"], torch.tensor([1.0]))


def test_aggregate_of_identical_states_is_identity():
    """Aggregating identical states returns that state."""
    state = {"w": torch.tensor([1.5, -2.5]), "b": torch.tensor([0.25])}
    out = aggregate(
        [ClientUpdate(i, 0, 10 * (i + 1), 0.0, state) for i in range(4)]
    )
    for k, v in state.items():
        torch.testing.assert_close(out[k], v)


def test_aggregate_preserves_integer_buffers():
    """Integer buffers are carried, not blended.

    Averaging an integer counter is meaningless.
    """
    a = {"n": torch.tensor([4], dtype=torch.int64)}
    b = {"n": torch.tensor([10], dtype=torch.int64)}
    out = aggregate(
        [ClientUpdate(0, 0, 1, 0.0, a), ClientUpdate(1, 0, 1, 0.0, b)]
    )
    assert out["n"].dtype == torch.int64
    assert out["n"].item() in (4, 10)


def test_aggregate_rejects_empty_and_zero_weight():
    """Aggregation rejects no updates and a zero total weight."""
    with pytest.raises(ValueError, match="no client updates"):
        aggregate([])
    with pytest.raises(ValueError, match="sum to zero"):
        aggregate([ClientUpdate(0, 0, 0, 0.0, {"w": torch.zeros(1)})])


# ── evaluation ────────────────────────────────────────────────────────────────


def test_evaluate_returns_sane_ranges():
    """Evaluation gives a positive loss and an accuracy in [0, 1]."""
    ds = tiny_dataset()
    model = build_model("small_cnn", ds.n_classes, seed=0)
    loss, acc = evaluate(model, ds.test_x, ds.test_y)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0


def test_evaluate_batching_does_not_change_the_answer():
    """Evaluation batch size does not change loss or accuracy."""
    ds = tiny_dataset()
    model = build_model("small_cnn", ds.n_classes, seed=0)
    a = evaluate(model, ds.test_x, ds.test_y, batch_size=7)
    b = evaluate(model, ds.test_x, ds.test_y, batch_size=512)
    assert a[0] == pytest.approx(b[0], rel=1e-5)
    assert a[1] == pytest.approx(b[1], rel=1e-9)


# ── the loop ──────────────────────────────────────────────────────────────────


def test_run_is_reproducible_from_seeds():
    """The load-bearing property of the whole harness.

    Detection probability and false-alarm rate are only interpretable if a
    configuration re-runs identically; without this, any measured difference
    could be run-to-run noise.
    """
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3, local_epochs=1, batch_size=32)
    parts = equal_parts(ds.n_train, 4)

    a = run_federated(cfg, ds, parts)
    b = run_federated(cfg, ds, parts)

    assert a.accuracy_curve() == b.accuracy_curve()
    for k, v in a.final_state.items():
        torch.testing.assert_close(v, b.final_state[k], rtol=0, atol=0)


def test_different_train_seed_changes_the_trajectory():
    """Asserted on parameters, not accuracy.

    Test accuracy over 80 samples moves in steps of 0.0125 and saturates on
    this fixture, so two genuinely different runs can report an identical
    curve. Parameters and training loss are the sensitive quantities.

    batch_size=16 on purpose: each client holds 60 samples, so the default
    batch of 64 is ONE batch per round and the train seed can only reorder
    samples inside it, which changes nothing but float summation order. CI on
    x86 caught exactly that, parameters one ulp apart and losses equal, and
    the Mac had passed by a rounding accident. With four batches per round
    the seed changes batch composition and the trajectories differ by ~1e-3
    on both architectures.
    """
    ds = tiny_dataset()
    parts = equal_parts(ds.n_train, 4)
    cfg_a = FedConfig(n_clients=4, rounds=3, train_seed=0, batch_size=16)
    cfg_b = FedConfig(n_clients=4, rounds=3, train_seed=99, batch_size=16)
    a = run_federated(cfg_a, ds, parts)
    b = run_federated(cfg_b, ds, parts)

    max_delta = max(
        float((a.final_state[k] - b.final_state[k]).abs().max())
        for k in a.final_state
        if torch.is_floating_point(a.final_state[k])
    )
    assert max_delta > 1e-5, max_delta
    loss_a = a.rounds[-1].mean_train_loss
    loss_b = b.rounds[-1].mean_train_loss
    assert abs(loss_a - loss_b) > 1e-6


def test_optimisation_reduces_training_loss():
    """The loop optimises: loss falls monotonically and the fixture is fit.

    Uses the default lr=0.01. At lr=0.05 with momentum=0.9 this fixture
    overshoots and training loss rebounds: an optimiser property, not a defect
    in the aggregation, but a reminder that a rising loss curve in a real run
    is a hyperparameter question before it is an integrity question.
    """
    ds = tiny_dataset()
    cfg = FedConfig(
        n_clients=4, rounds=15, local_epochs=1, batch_size=32, lr=0.01
    )
    result = run_federated(cfg, ds, equal_parts(ds.n_train, 4))

    losses = [r.mean_train_loss for r in result.rounds]
    assert all(
        losses[i] > losses[i + 1] for i in range(len(losses) - 1)
    ), losses
    # Measured at 0.36 of the initial loss over 15 rounds; 0.6 leaves headroom
    # for platform float variation without becoming a vacuous assertion.
    assert losses[-1] < 0.6 * losses[0], losses
    assert result.final_acc > 0.9


def test_history_shape_and_participation():
    """Run history has one record per round, each with every client."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=5)
    result = run_federated(cfg, ds, equal_parts(ds.n_train, 4))
    assert len(result.rounds) == 5
    assert [r.round_idx for r in result.rounds] == [0, 1, 2, 3, 4]
    for r in result.rounds:
        assert r.participants == [0, 1, 2, 3]


def test_partial_participation_samples_a_subset():
    """client_fraction=0.5 samples two of four clients each round."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=4, client_fraction=0.5)
    result = run_federated(cfg, ds, equal_parts(ds.n_train, 4))
    for r in result.rounds:
        assert len(r.participants) == 2
        assert set(r.participants) <= {0, 1, 2, 3}


def test_update_hook_sees_every_client_before_aggregation():
    """This hook is the seam the integrity layer attaches to."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    seen: list[tuple[int, list[int]]] = []

    run_federated(
        cfg,
        ds,
        equal_parts(ds.n_train, 4),
        update_hook=lambda ups, rnd: seen.append(
            (rnd, [u.client_id for u in ups])
        ),
    )

    assert seen == [(0, [0, 1, 2, 3]), (1, [0, 1, 2, 3]), (2, [0, 1, 2, 3])]


def test_update_hook_does_not_alter_results():
    """The baseline must be identical with and without instrumentation."""
    ds = tiny_dataset()
    cfg = FedConfig(n_clients=4, rounds=3)
    parts = equal_parts(ds.n_train, 4)
    without = run_federated(cfg, ds, parts)
    with_hook = run_federated(
        cfg, ds, parts, update_hook=lambda ups, rnd: None
    )
    assert without.accuracy_curve() == with_hook.accuracy_curve()


def test_partition_count_mismatch_raises():
    """Fewer partitions than clients raises ValueError."""
    ds = tiny_dataset()
    with pytest.raises(ValueError, match="partitions for"):
        run_federated(
            FedConfig(n_clients=4, rounds=1), ds, equal_parts(ds.n_train, 3)
        )


# ── config ────────────────────────────────────────────────────────────────────


def test_config_rejects_invalid_values():
    """FedConfig rejects out-of-range values."""
    with pytest.raises(ValueError, match="n_clients"):
        FedConfig(n_clients=1)
    with pytest.raises(ValueError, match="client_fraction"):
        FedConfig(client_fraction=0.0)
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        FedConfig(dirichlet_alpha=0.0)
    with pytest.raises(ValueError, match="rounds"):
        FedConfig(rounds=0)


def test_fingerprint_tracks_config_changes():
    """Config fingerprint is stable and changes when a field changes."""
    base = FedConfig()
    assert base.fingerprint() == FedConfig().fingerprint()
    assert base.fingerprint() != FedConfig(dirichlet_alpha=0.6).fingerprint()
    assert base.fingerprint() != FedConfig(train_seed=1).fingerprint()
