"""Shared fixtures for the suite. Plain functions, imported explicitly."""

from __future__ import annotations

import numpy as np
import torch

from telltale.data import Dataset


def tiny_dataset(
    n_train: int = 240, n_test: int = 80, n_classes: int = 4
) -> Dataset:
    """Small synthetic set so the suite runs in seconds without a download.

    Linearly separable by construction: each class gets a constant offset on a
    distinct band of rows. That is deliberate. These tests verify that the
    federated loop *optimises*: that gradients flow, aggregation composes, and
    seeds reproduce. They do not and cannot verify generalisation: 60 samples
    per client against a 206k-parameter CNN memorises rather than learns.

    Generalisation is demonstrated by the real FashionMNIST baseline in
    scripts/run_baseline.py, which is where that claim belongs.
    """
    g = torch.Generator().manual_seed(0)
    ytr = torch.arange(n_classes).repeat(n_train // n_classes)
    yte = torch.arange(n_classes).repeat(n_test // n_classes)

    def make(y: torch.Tensor) -> torch.Tensor:
        base = torch.randn(len(y), 1, 28, 28, generator=g) * 0.3
        for k in range(n_classes):
            base[y == k, :, k * 5 : k * 5 + 4, :] += 2.0
        return base

    return Dataset(
        make(ytr), ytr, make(yte), yte, n_classes=n_classes, name="synthetic"
    )


def equal_parts(n: int, k: int) -> list[np.ndarray]:
    """Split range(n) into k strided index arrays of near-equal size."""
    return [np.arange(i, n, k) for i in range(k)]
