from __future__ import annotations

import numpy as np
import pytest
import torch

from telltale.data import (
    _assert_exact_partition,
    dirichlet_partition,
    load_dataset,
    partition_label_matrix,
    skew_summary,
)


def make_labels(n_per_class: int = 200, n_classes: int = 10) -> torch.Tensor:
    """Return balanced labels, n_per_class of each class in order."""
    return torch.arange(n_classes).repeat_interleave(n_per_class)


def test_partition_is_exact_and_disjoint():
    """Every sample lands with exactly one client.

    Silent data loss here would bias every downstream measurement, so it is
    asserted rather than assumed.
    """
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=8, alpha=0.5, seed=0)

    joined = np.concatenate(parts)
    assert joined.size == len(y)
    assert np.unique(joined).size == len(y)
    assert set(joined.tolist()) == set(range(len(y)))


def test_partition_respects_minimum_client_size():
    """No client receives fewer than min_client_samples."""
    y = make_labels()
    parts = dirichlet_partition(
        y, n_clients=8, alpha=0.1, seed=3, min_client_samples=25
    )
    assert min(p.size for p in parts) >= 25


def test_partition_is_deterministic_given_seed():
    """The same seed reproduces the same partition."""
    y = make_labels()
    a = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=42)
    b = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=42)
    # Both partitions have n_clients=6 parts, so the lengths match.
    for pa, pb in zip(a, b, strict=True):
        np.testing.assert_array_equal(pa, pb)


def test_different_seeds_give_different_partitions():
    """Different seeds give a different partition."""
    y = make_labels()
    a = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=1)
    b = dirichlet_partition(y, n_clients=6, alpha=0.5, seed=2)
    # Both partitions have n_clients=6 parts, so the lengths match.
    assert any(
        pa.size != pb.size or not np.array_equal(pa, pb)
        for pa, pb in zip(a, b, strict=True)
    )


def test_low_alpha_is_more_skewed_than_high_alpha():
    """The knob has to actually do what the study claims it does.

    Averaged over seeds, because a single Dirichlet draw is noisy enough that
    one low-alpha partition can come out milder than one high-alpha partition.
    """
    y = make_labels()

    def mean_skew(alpha: float) -> float:
        vals = []
        for seed in range(8):
            parts = dirichlet_partition(y, n_clients=8, alpha=alpha, seed=seed)
            m = partition_label_matrix(y, parts, n_classes=10)
            vals.append(skew_summary(m)["mean_tv_from_uniform"])
        return float(np.mean(vals))

    assert mean_skew(0.1) > mean_skew(1.0) > mean_skew(100.0)


def test_near_iid_alpha_approaches_uniform():
    """A very large alpha gives near-uniform client label mixes."""
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=8, alpha=500.0, seed=0)
    m = partition_label_matrix(y, parts, n_classes=10)
    assert skew_summary(m)["mean_tv_from_uniform"] < 0.10


def test_label_matrix_totals_match_partition_sizes():
    """Label-count matrix rows sum to each client's partition size."""
    y = make_labels()
    parts = dirichlet_partition(y, n_clients=5, alpha=0.7, seed=11)
    m = partition_label_matrix(y, parts, n_classes=10)
    assert m.sum() == len(y)
    for i, p in enumerate(parts):
        assert m[i].sum() == p.size


def test_impossible_minimum_raises():
    """A minimum client size above the data total raises ValueError."""
    y = make_labels(n_per_class=2, n_classes=10)  # 20 samples total
    with pytest.raises(ValueError, match="cannot give"):
        dirichlet_partition(
            y, n_clients=10, alpha=0.5, seed=0, min_client_samples=50
        )


def test_unsatisfiable_partition_raises_after_retries():
    """An unlucky-every-time partition raises RuntimeError after max_tries."""
    y = make_labels(n_per_class=12, n_classes=10)  # 120 samples, 20 clients
    with pytest.raises(RuntimeError, match="could not draw a partition"):
        dirichlet_partition(
            y,
            n_clients=20,
            alpha=0.01,
            seed=0,
            min_client_samples=5,
            max_tries=5,
        )


def test_partition_check_catches_dropped_samples():
    """No public input can trigger the guard, so it is tested directly."""
    with pytest.raises(AssertionError, match="covers 5 of 6"):
        _assert_exact_partition(
            [np.array([0, 1]), np.array([2, 3, 4])], total=6
        )


def test_partition_check_catches_a_sample_given_to_two_clients():
    """The partition guard rejects a sample assigned to two clients."""
    with pytest.raises(AssertionError, match="more than one client"):
        _assert_exact_partition(
            [np.array([0, 1, 2]), np.array([2, 3, 4])], total=6
        )


# ── load_dataset, with no download ────────────────────────────────────────────


def fake_torchvision_dataset(calls: list[dict]):
    """Stand-in for a torchvision dataset class: uint8 pixels, no network."""

    class Fake:
        def __init__(self, root, train, download):
            calls.append({"root": root, "train": train, "download": download})
            g = torch.Generator().manual_seed(1 if train else 2)
            n = 30 if train else 10
            self.data = torch.randint(
                0, 256, (n, 28, 28), generator=g, dtype=torch.uint8
            )
            self.targets = torch.arange(n) % 10

    return Fake


@pytest.mark.parametrize(
    ("name", "attr", "mean", "std"),
    [
        ("fashion_mnist", "FashionMNIST", 0.2860, 0.3530),
        ("mnist", "MNIST", 0.1307, 0.3081),
    ],
)
def test_load_dataset_normalises_with_the_published_statistics(
    tmp_path, monkeypatch, name, attr, mean, std
):
    """Pixels are scaled to [0, 1], then standardised.

    Both splits use the same training-split statistics, so the test split
    never informs the scale.
    """
    pytest.importorskip("torchvision")
    calls: list[dict] = []
    monkeypatch.setattr(
        f"torchvision.datasets.{attr}", fake_torchvision_dataset(calls)
    )

    root = tmp_path / "nested" / "data"
    ds = load_dataset(name, root=root)

    assert root.is_dir()
    assert [c["train"] for c in calls] == [True, False]
    assert all(c["download"] and c["root"] == str(root) for c in calls)

    raw = fake_torchvision_dataset([])(root, True, False).data
    expected = ((raw.to(torch.float32) / 255.0 - mean) / std).unsqueeze(1)
    assert (
        ds.train_x.shape == (30, 1, 28, 28)
        and ds.test_x.shape == (10, 1, 28, 28)
    )
    torch.testing.assert_close(ds.train_x, expected)
    assert ds.train_x.dtype == torch.float32 and ds.train_y.dtype == torch.int64
    assert ds.n_classes == 10 and ds.name == name and ds.n_train == 30


def test_load_dataset_rejects_an_unknown_name(tmp_path):
    """An unknown dataset name raises ValueError."""
    pytest.importorskip("torchvision")
    with pytest.raises(ValueError, match="unknown dataset 'cifar'"):
        load_dataset("cifar", root=tmp_path)


def test_load_dataset_delegates_deepship_to_the_acoustic_loader(
    tmp_path, monkeypatch
):
    """deepship is routed to the acoustic loader with the expected paths."""
    seen = {}
    sentinel = object()

    def fake_loader(root, cache):
        seen.update(root=root, cache=cache)
        return sentinel

    monkeypatch.setattr("telltale.acoustic.load_deepship", fake_loader)

    assert load_dataset("deepship", root=tmp_path) is sentinel
    assert seen == {
        "root": tmp_path / "deepship",
        "cache": tmp_path / "deepship_28x28.pt",
    }
