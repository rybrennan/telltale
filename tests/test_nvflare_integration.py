"""The integrity layer as an NVFlare ModelAggregator; no simulator, no network.

`reset_stats` / `accept_model` / `aggregate_model` are called in the order
NVFlare's FedAvg controller calls them, with FLModel results shaped exactly as
examples/nvflare/client.py sends them.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

pytest.importorskip("nvflare")
from nvflare.apis.fl_constant import FLMetaKey  # noqa: E402
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType  # noqa: E402

from telltale.integrations.nvflare import (  # noqa: E402
    FINGERPRINT_KEY,
    FLAGGED_KEY,
    PROBE_KEY,
    IntegrityAggregator,
    probe_meta,
)

K, P, C, R = 5, 20, 10, 30
FP = "battery-abc"


def result(
    site: int, probs: np.ndarray, value: float, steps: int = 100, fp: str = FP
) -> FLModel:
    """One client's FLModel: one tensor filled with `value`, plus scores."""
    meta = {
        "client_name": f"site-{site}",
        FLMetaKey.NUM_STEPS_CURRENT_ROUND: steps,
    }
    meta.update(probe_meta(probs, fp))
    return FLModel(
        params={"w": np.full((3,), value, dtype=np.float32)},
        params_type=ParamsType.FULL,
        meta=meta,
    )


def aggregator(**kw) -> IntegrityAggregator:
    return IntegrityAggregator(
        n_probes=P,
        n_classes=C,
        battery_fingerprint=FP,
        num_rounds=R,
        max_clients=K,
        **kw,
    )


def run_round(agg: IntegrityAggregator, results: list[FLModel]) -> FLModel:
    """One FedAvg round as the controller drives it."""
    agg.reset_stats()
    for m in results:
        agg.accept_model(m)
    return agg.aggregate_model()


def uniform() -> np.ndarray:
    return np.full((P, C), 1.0 / C)


def round_probs(
    base: np.ndarray, rng: np.random.Generator, r: int, drift_row: int | None
) -> np.ndarray:
    """Stable per-client preference plus noise; one row drifts from round 15."""
    out = base + rng.normal(0, 0.01, base.shape)
    if drift_row is not None and r >= 15:
        a = min(1.0, (r - 15 + 1) / 10) * 0.6
        out[drift_row] = (1 - a) * out[drift_row] + a * np.eye(C)[0]
    out = np.clip(out, 1e-4, None)
    return out / out.sum(-1, keepdims=True)


def drive(
    agg: IntegrityAggregator, drift_row: int | None, seed: int, sampled=None
) -> None:
    """R rounds; `sampled(r)` gives round r's reporting rows (default all)."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(99).dirichlet(np.ones(C), size=(K, P))
    for r in range(R):
        probs = round_probs(base, rng, r, drift_row)
        rows = range(K) if sampled is None else sampled(r)
        run_round(agg, [result(k + 1, probs[k], float(k)) for k in rows])


def test_payload_is_float16_and_about_one_kilobyte():
    p = (
        np.random.default_rng(0)
        .dirichlet(np.ones(C), size=50)
        .astype(np.float32)
    )
    meta = probe_meta(p, FP)
    assert len(meta[PROBE_KEY]) == 50 * C * 2
    assert meta[FINGERPRINT_KEY] == FP


def test_every_constructor_argument_survives_job_export():
    """NVFlare exports a component by reading its constructor args back."""
    agg = aggregator(
        reference_start=3,
        reference_stop=9,
        threshold=1.5,
        exclude_flagged=True,
        history_path="/x",
    )
    for name in inspect.signature(IntegrityAggregator.__init__).parameters:
        if name != "self":
            assert hasattr(agg, name), name
    assert agg.reference == slice(3, 9)


def test_weighted_average_uses_num_steps():
    agg = aggregator()
    out = run_round(
        agg,
        [
            result(1, uniform(), 1.0, steps=100),
            result(2, uniform(), 4.0, steps=300),
        ],
    )
    np.testing.assert_allclose(out.params["w"], (1.0 * 100 + 4.0 * 300) / 400)
    assert out.params_type == ParamsType.FULL
    assert out.meta[FLAGGED_KEY] == []


def test_site_k_is_pinned_to_row_k_minus_one_whatever_the_arrival_order():
    agg = aggregator()
    run_round(agg, [result(4, uniform(), 0.0), result(2, uniform(), 0.0)])
    assert agg.client_index == {"site-4": 3, "site-2": 1}


def test_fingerprint_mismatch_and_missing_scores_are_recorded_not_scored():
    agg = aggregator()
    no_scores = FLModel(
        params={"w": np.zeros(3)},
        params_type=ParamsType.FULL,
        meta={"client_name": "site-3"},
    )
    run_round(
        agg,
        [
            result(1, uniform(), 0.0),
            result(2, uniform(), 0.0, fp="WRONG"),
            no_scores,
        ],
    )
    assert agg.fingerprint_mismatches == [(0, "site-2")]
    assert agg.missing_scores == [(0, "site-3")]
    assert set(agg.probs[0]) == {0}


def test_exclusion_is_off_by_default():
    agg = aggregator()
    agg._ever_flagged = {"site-5"}
    out = run_round(agg, [result(k + 1, uniform(), float(k)) for k in range(K)])
    np.testing.assert_allclose(out.params["w"], 2.0)


def test_exclude_flagged_leaves_the_flagged_client_out_of_the_average():
    agg = aggregator(exclude_flagged=True)
    agg._ever_flagged = {"site-5"}
    out = run_round(agg, [result(k + 1, uniform(), float(k)) for k in range(K)])
    np.testing.assert_allclose(
        out.params["w"], 1.5
    )  # rows 0-3 average to 1.5; row 4 (4.0) excluded
    assert agg.excluded[0] == ["site-5"]


def test_exclusion_never_aggregates_nothing():
    agg = aggregator(exclude_flagged=True)
    agg._ever_flagged = {"site-1"}
    out = run_round(agg, [result(1, uniform(), 7.0)])
    np.testing.assert_allclose(out.params["w"], 7.0)


def test_detector_flags_the_drifting_client_first():
    healthy = aggregator()
    drive(healthy, None, seed=1)
    thr = float(np.nanmax(healthy.cusum_so_far()[15:]))

    faulted = aggregator(threshold=thr)
    drive(faulted, drift_row=2, seed=2)
    first = min(r for r, names in faulted.flagged.items() if names)
    assert first >= 15
    assert faulted.flagged[first] == ["site-3"]


def test_partial_participation_scores_only_reporting_clients():
    """An unsampled client is absent (NaN), not zero; CUSUM uses contacts."""
    agg = aggregator()
    drive(
        agg,
        None,
        seed=3,
        sampled=lambda r: [k for k in range(K) if (r + k) % 5 != 0],
    )  # 4 of 5 each round
    arr = agg.probs_array()
    assert np.isnan(arr[0, 0]).all() and not np.isnan(arr[0, 1]).any()
    c = agg.cusum_so_far()
    assert np.isnan(c[:15]).all()
    assert np.isfinite(c[15:]).sum() == 4 * (
        R - 15
    )  # one finite value per contact after the reference


def test_run_shorter_than_the_reference_epoch_records_without_failing():
    agg = IntegrityAggregator(
        n_probes=P,
        n_classes=C,
        battery_fingerprint=FP,
        num_rounds=3,
        max_clients=K,
    )
    for _ in range(3):
        run_round(agg, [result(k + 1, uniform(), 0.0) for k in range(K)])
    assert np.isnan(agg.cusum_so_far()).all()
    assert len(agg.probs) == 3
