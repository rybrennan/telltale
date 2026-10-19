"""Run-level false alarms, pooled calibration and the strict detection rule."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import review4_scores as r4  # noqa: E402
import score_faults as sf  # noqa: E402

CLASSES = np.array(["ship"] * 4)


def healthy(seed):
    return np.random.default_rng(seed).normal(size=(30, 4))


def test_leave_one_out_run_alarms_are_fixed_by_construction():
    """With distinct maxima, exactly one of n+1 runs exceeds the maximum of
    the other n, and half of all calibrate-one / hold-one pairs alarm. The
    run-level rate is 1/(n+1) by arithmetic, not a measurement."""
    runs = [healthy(s) for s in range(6)]
    pooled = []
    for s in range(6):
        thr = r4.pooled_threshold(
            [runs[h] for h in range(6) if h != s], CLASSES, "x"
        )
        pooled.append(r4.run_alarm(sf.flags_by_class(runs[s], thr, CLASSES)))
    assert sum(pooled) == 1
    single = [
        r4.run_alarm(
            sf.flags_by_class(
                runs[h],
                sf.calibrate_by_class(runs[c], CLASSES, "x", sf.BURN_IN),
                CLASSES,
            )
        )
        for c in range(6)
        for h in range(6)
        if h != c
    ]
    assert sum(single) == 15


def test_strict_rule_needs_the_target_first():
    fl = np.zeros((30, 4), dtype=bool)
    fl[18, 2] = True
    clean = r4.fault_outcome(fl, target=2, onset=15)
    assert clean["detected"] and clean["strict"] and clean["false_clients"] == 0
    fl[12, 0] = True  # a healthy client flagged before onset
    early = r4.fault_outcome(fl, target=2, onset=15)
    assert early["detected"] and not early["strict"]
    assert early["false_clients"] == 1
    fl2 = np.zeros((30, 4), dtype=bool)
    fl2[18, 2] = True
    fl2[25, 1] = fl2[26, 1] = fl2[27, 3] = True  # after detection
    late = r4.fault_outcome(fl2, target=2, onset=15)
    assert late["strict"] and late["false_clients"] == 2
    assert late["false_node_rounds"] == 3


def test_main_grid_filter_excludes_the_review4_cells():
    """The onset cells share the main grid's fingerprint, so only the fault's
    onset tells them apart; another partition or clipping is in the config."""
    main = {"config": {"partition_seed": 0}, "fault": {"onset": 15}}
    healthy = {"config": {"partition_seed": 0}}
    assert sf.in_main_grid(main) and sf.in_main_grid(healthy)
    assert not sf.in_main_grid(
        {"config": {"partition_seed": 0}, "fault": {"onset": 10}}
    )
    assert not sf.in_main_grid(
        {"config": {"partition_seed": 0}, "fault": {"onset": 22}}
    )
    assert not sf.in_main_grid({"config": {"partition_seed": 2}})
    assert not sf.in_main_grid(
        {"config": {"partition_seed": 0, "clip_norm": 0.5}, "fault": None}
    )
