from __future__ import annotations

import numpy as np
import pytest

from telltale.metrics import (
    Threshold,
    calibrate,
    calibrate_by_class,
    detection_probability,
    false_alarm_rate,
    flags,
    flags_by_class,
    score_fault_run,
    score_flags,
)


def healthy_series(R=20, K=5, seed=0):
    """Return a uniform-noise healthy score series with NaN early rounds."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.2, 0.5, size=(R, K))
    s[:3] = np.nan  # early rounds undefined, as persistence_series produces
    return s


def test_calibrate_max_and_quantile_ignore_burn_in_and_nan():
    """Max and quantile calibration skip burn-in rounds and NaNs."""
    h = healthy_series()
    h[4, 1] = 5.0  # inside burn-in: must not set the threshold
    thr = calibrate(h, "persistence", burn_in=5)
    assert thr.method == "max"
    assert thr.value == pytest.approx(np.nanmax(h[5:]))
    assert thr.value < 5.0
    q = calibrate(h, "persistence", burn_in=5, quantile=0.9)
    assert q.value < thr.value
    assert q.method == "quantile:0.9"
    with pytest.raises(ValueError, match="burn_in"):
        calibrate(h, "persistence", burn_in=20)
    with pytest.raises(ValueError, match="no finite"):
        calibrate(np.full((6, 2), np.nan), "x", burn_in=1)


def test_flags_respect_burn_in_and_nan():
    """No flag is raised inside burn-in or on a NaN score."""
    s = np.array([[np.nan, 1.0], [0.9, 1.0], [0.1, 1.0]])
    thr = Threshold("x", 0.5, burn_in=1, method="max")
    f = flags(s, thr)
    assert f.tolist() == [[False, False], [True, True], [False, True]]


def test_false_alarm_rate_is_zero_on_the_calibration_run_by_construction():
    """Documents the reason FAR must be measured on a different seed."""
    h = healthy_series()
    thr = calibrate(h, "persistence", burn_in=5)
    assert false_alarm_rate(h, thr) == 0.0
    other = healthy_series(seed=1)
    far = false_alarm_rate(other, thr)
    assert 0.0 <= far <= 1.0


def test_score_fault_run_hit_miss_and_timing():
    """Fault scoring gets hits, misses, timing and false localisations right."""
    R, K, target, onset = 20, 5, 2, 10
    thr = Threshold("persistence", 0.6, burn_in=5, method="max")
    s = np.full((R, K), 0.3)
    s[13, target] = 0.9  # first exceedance 3 rounds after onset
    s[15, target] = 0.9
    s[7, 4] = 0.95  # a healthy node flagged after burn-in
    s[2, target] = 0.99  # inside burn-in: ignored
    out = score_fault_run(s, thr, target, onset)
    assert (
        out.detected
        and out.detection_round == 13
        and out.time_to_detection == 3
    )
    assert out.target_flag_count == 2
    assert out.false_localisations == 1

    miss = score_fault_run(np.full((R, K), 0.3), thr, target, onset)
    assert not miss.detected and miss.time_to_detection is None

    # A flag on the target before onset does not count as a detection.
    early = np.full((R, K), 0.3)
    early[8, target] = 0.9
    assert not score_fault_run(early, thr, target, onset).detected
    with pytest.raises(ValueError):
        score_fault_run(s, thr, 9, onset)


def test_detection_probability():
    """Detection probability is the hit fraction, and NaN with no runs."""
    thr = Threshold("x", 0.5, 0, "max")
    a = score_fault_run(np.full((10, 2), 0.9), thr, 0, 3)
    b = score_fault_run(np.full((10, 2), 0.1), thr, 0, 3)
    assert detection_probability([a, b]) == 0.5
    assert np.isnan(detection_probability([]))


def test_per_class_thresholds_judge_each_node_by_its_own_class():
    """Per-class thresholds judge each node against its own class ceiling."""
    h = healthy_series(K=4)
    # Class B node runs hot in health; class A must not inherit its ceiling.
    h[5:, 3] += 10.0
    classes = np.array(["A", "A", "A", "B"])
    thr = calibrate_by_class(h, classes, "x", burn_in=5)
    assert thr["B"].value > thr["A"].value + 5
    assert thr["A"].value == pytest.approx(calibrate(h[:, :3], "x", 5).value)
    s = np.full(h.shape, 0.0)
    s[10, 0] = thr["A"].value + 0.01  # above A's ceiling, below B's
    s[10, 3] = thr["A"].value + 0.01  # same value on a B node: not a flag
    f = flags_by_class(s, thr, classes)
    assert f[10, 0] and not f[10, 3]
    out = score_flags(f, 5, target=0, onset=8)
    assert (
        out.detected
        and out.detection_round == 10
        and out.false_localisations == 0
    )
