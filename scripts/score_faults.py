#!/usr/bin/env python3
"""Stage 4: score every fault run against thresholds calibrated on healthy runs.

For each alpha with a healthy probes_* run, calibrate one threshold per
statistic at the healthy maximum after burn-in, then score every fault_* run
at that alpha. Prints one row per (fault run, statistic):

    detected, round detected, time to detection, false localisations,
    fleet-accuracy delta (what the aggregate metric would have shown)

Thresholds never see a fault. With one healthy seed the false-alarm rate on
the calibration run is zero by construction and is NOT reported as a FAR; a
FAR needs healthy runs on other train seeds, which the grid must supply.

    python scripts/score_faults.py

Runs that carry a .updates.npz sidecar (run_grid.py --update-stats) are also
scored on four update-space baselines (telltale/baselines.py): FedRR's
residual-rank CUSUM and its update-norm benchmark, and the update norm and
FedCE's leave-one-out update distance under telltale's own CUSUM. A statistic
is scored only where every run involved has it. FedRR is also scored at its
own control limit (in-control ARL 30, as in the paper), next to the
healthy-max threshold every other statistic uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from telltale.baselines import (  # noqa: E402
    FIT_ROUNDS,
    fedrr_cusum,
    fedrr_threshold,
    local_steps,
)
from telltale.contact import ContactSchedule  # noqa: E402
from telltale.detect import (  # noqa: E402
    cusum,
    level_series,
    offset_change,
    persistence_series,
    self_divergence,
    self_referenced_level,
)
from telltale.metrics import (  # noqa: E402
    Threshold,
    calibrate_by_class,
    flags_by_class,
    node_round_rate,
    round_alarm_rate,
    score_flags,
)

RESULTS = REPO / "results"
BURN_IN = 10
WINDOW, SPAN = 5, 5
REFERENCE = slice(5, 15)
"""CUSUM reference epoch: rounds 5-14, the settled part of the burn-in.

The same for every node and every run. Faults in this study start at round 15.
"""


def platform_classes(d: dict) -> np.ndarray:
    """Label each node 'sub' if it is on a contact schedule, else 'ship'.

    Thresholds are calibrated per class and CUSUM's reference SD is pooled
    within a class.
    """
    K = d["config"]["n_clients"]
    subs = set((d.get("contact") or {}).get("sub_nodes", []))
    return np.array(["sub" if k in subs else "ship" for k in range(K)])


FEDRR_START = FIT_ROUNDS[-1] + 1
"""First round FedRR monitors: the one after its start-up fit."""
FEDRR_D = 0.5
"""FedRR's CUSUM reference value; the middle of the paper's 0.4 to 0.6."""
FEDRR_ARL0 = 30
"""FedRR's in-control average run length, as in the paper's experiments."""


def statistics(
    probs: np.ndarray | dict,
    classes: np.ndarray,
    updates: dict | None = None,
    seed: int = 0,
    steps: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute every scored statistic as a [round, node] array.

    Args:
        probs: Probe scores, shape [round, node, probe, class].
        classes: Platform class per node, from platform_classes.
        updates: The run's .updates.npz contents, or None.
        seed: The run's train seed; seeds FedRR's rank jitter.
        steps: (clients,) local SGD steps per round, for the step-normalised
            FedRR variants; omitted when None.

    Returns:
        Map from statistic name (the names in STATS) to its series. The
        update-space baselines are present only when `updates` is given.
    """
    base = {}
    if updates is not None:
        base = {
            "update_norm_cusum": cusum(
                updates["norm"], REFERENCE, pool=classes
            ),
            "update_loo_cusum": cusum(
                updates["loo_dist"], REFERENCE, pool=classes
            ),
            "fedrr": fedrr_cusum(
                updates["resid"], FEDRR_START, FEDRR_D, seed=seed
            ),
            "fedrr_norm": fedrr_cusum(
                updates["norm"], FEDRR_START, FEDRR_D, seed=seed
            ),
        }
        if "norm_preclip" in updates:
            # The client's own update norm before clipping, as a client
            # could self-report it; differs from update_norm_cusum only in
            # runs that clip.
            base["update_preclip_cusum"] = cusum(
                updates["norm_preclip"], REFERENCE, pool=classes
            )
        if steps is not None:
            base["fedrr_steps"] = fedrr_cusum(
                updates["resid"] / steps, FEDRR_START, FEDRR_D, seed=seed
            )
            base["fedrr_norm_steps"] = fedrr_cusum(
                updates["norm"] / steps, FEDRR_START, FEDRR_D, seed=seed
            )
    return base | probe_statistics(probs, classes)


def probe_statistics(
    probs: np.ndarray | dict, classes: np.ndarray
) -> dict[str, np.ndarray]:
    """The probe-score statistics of one run, each a [round, node] array.

    Every one depends only on the run itself, so a machine without the
    probe array can score a run from these (scripts/export_stats.py writes
    them to <stem>.stats.npz).

    Args:
        probs: Probe scores [round, node, probe, class], or a dict of
            already computed statistics, returned unchanged.
        classes: Platform class per node.

    Returns:
        Map from probe statistic name to its series.
    """
    if isinstance(probs, dict):
        return probs
    lv = level_series(probs)
    return {
        "level": lv["level"],
        "level_z": np.abs(lv["z"]),
        "self": self_divergence(probs),
        "persistence": persistence_series(
            offset_change(lv["offset"], WINDOW), SPAN
        ),
        "level_own_z": self_referenced_level(lv["level"], WINDOW),
        "level_cusum": cusum(lv["level"], REFERENCE, pool=classes),
        "level_cusum2": cusum(
            lv["level"], REFERENCE, pool=classes, two_sided=True
        ),
    }


def load(stem: str) -> tuple[dict, np.ndarray | dict, dict | None]:
    """Read a result's JSON, probe scores and update sidecar from results/.

    The precomputed statistics in <stem>.stats.npz are used when present;
    the raw probe array only when they are absent.
    """
    side = RESULTS / f"{stem}.updates.npz"
    updates = None
    if side.exists():
        with np.load(side) as z:
            updates = {k: z[k] for k in z.files}
    stats = RESULTS / f"{stem}.stats.npz"
    if stats.exists():
        # The exported statistics are the published record. A raw array left
        # over from another machine's run of the same configuration must not
        # silently replace them.
        with np.load(stats) as z:
            probs = {k: z[k] for k in z.files}
    else:
        probs = np.load(RESULTS / f"{stem}.npz")["probs"]
    return json.loads((RESULTS / f"{stem}.json").read_text()), probs, updates


def scored(d: dict, probs: np.ndarray, updates: dict | None) -> dict:
    """Every statistic for one run, seeded by its train seed."""
    cfg = d["config"]
    steps = local_steps(
        d["partition_summary"]["client_sizes"],
        cfg["batch_size"],
        cfg["local_epochs"],
    )
    return statistics(
        probs, platform_classes(d), updates, cfg["train_seed"], steps
    )


def family(d: dict) -> tuple:
    """Key a run by everything that identifies it but its train seed and tag.

    Healthy runs in one family differ only by train_seed; the lowest seed is
    the calibration run and the others are held out for the false-alarm rate.
    """
    cfg = {
        k: v
        for k, v in d["config"].items()
        if k not in ("train_seed", "tag", "notes", "extra")
    }
    return (
        json.dumps(cfg, sort_keys=True),
        d["probe_fingerprint"],
        json.dumps(d.get("contact"), sort_keys=True),
    )


def observed_mask(d: dict) -> np.ndarray | None:
    """Return the [round, node] observed mask, or None with no schedule."""
    c = d.get("contact")
    if not c:
        return None
    return ContactSchedule(
        period=c["period"], sub_nodes=tuple(c["sub_nodes"])
    ).observed(d["config"]["rounds"], d["config"]["n_clients"])


STATS = (
    "level",
    "level_z",
    "self",
    "persistence",
    "level_own_z",
    "level_cusum",
    "level_cusum2",
    "update_norm_cusum",
    "update_loo_cusum",
    "update_preclip_cusum",
    "fedrr",
    "fedrr_norm",
    "fedrr_steps",
    "fedrr_norm_steps",
)
QUANTILES = (None, 0.99, 0.95, 0.9)
"""Threshold sweep: healthy max, then quantiles of the healthy node-rounds."""


def main() -> int:
    """Print the false-alarm, detection and sweep tables; write scores.json."""
    families: dict[tuple, dict[int, tuple[dict, dict]]] = {}
    for f in sorted(RESULTS.glob("probes_*.json")):
        d, probs, upd = load(f.stem)
        families.setdefault(family(d), {})[d["config"]["train_seed"]] = (
            d,
            scored(d, probs, upd),
        )
    faults = []
    for f in sorted(RESULTS.glob("fault_*.json")):
        d, probs, upd = load(f.stem)
        faults.append((d, scored(d, probs, upd)))

    # ── false-alarm rate on held-out healthy seeds ──
    print(
        "## False alarms: calibrate on the lowest train seed, count on the "
        f"others (burn-in {BURN_IN})\n"
    )
    print(
        "| dataset | alpha | calib seed | held-out seeds | stat | thr | "
        "node-round FAR | rounds with any alarm |"
    )
    print("|---|---|---|---|---|---|---|---|")
    far: dict[tuple, float] = {}
    summary: dict = {"far": [], "pd": [], "sweep": []}
    for fam, runs in sorted(
        families.items(),
        key=lambda kv: next(iter(kv[1].values()))[0]["config"][
            "dirichlet_alpha"
        ],
    ):
        seeds = sorted(runs)
        calib_d, calib_stats = runs[seeds[0]]
        held = [runs[s] for s in seeds[1:]]
        alpha = calib_d["config"]["dirichlet_alpha"]
        if not held:
            print(
                f"| {calib_d['config']['dataset']} | {alpha} | {seeds[0]} | "
                "none yet | - | - | - | - |"
            )
            continue
        classes = platform_classes(calib_d)
        for name in STATS:
            if name not in calib_stats or any(name not in s for _, s in held):
                continue
            thr = calibrate_by_class(calib_stats[name], classes, name, BURN_IN)
            node_rounds = []
            any_alarm = []
            for _, st in held:
                fl = flags_by_class(st[name], thr, classes)
                node_rounds.append(node_round_rate(fl, st[name], BURN_IN))
                any_alarm.append(round_alarm_rate(fl, st[name], BURN_IN))
            far[(fam, name)] = float(np.mean(node_rounds))
            summary["far"].append(
                {
                    "dataset": calib_d["config"]["dataset"],
                    "alpha": alpha,
                    "contact": bool(calib_d.get("contact")),
                    "stat": name,
                    "thresholds": {c: t.value for c, t in thr.items()},
                    "node_round_far": float(np.mean(node_rounds)),
                    "any_alarm": float(np.mean(any_alarm)),
                }
            )
            thr_txt = " / ".join(
                f"{c}:{t.value:.3f} (n={t.n_calibration})"
                for c, t in thr.items()
            )
            print(
                f"| {calib_d['config']['dataset']} | {alpha} | {seeds[0]} | "
                f"{seeds[1:]} | {name} | {thr_txt} | "
                f"{np.mean(node_rounds):.3f} | {np.mean(any_alarm):.3f} |"
            )

    if not faults:
        print("\nno fault_*.json in results/")
        return 0
    # Thresholds come from each family's calibration seed and are applied to
    # fault runs on EVERY seed of that family; Pd at a grid point is then the
    # fraction of seeds detected.
    healthy = {fam: runs[min(runs)] for fam, runs in families.items()}
    print()

    print(
        "## Fault runs (CUSUM reference rounds "
        f"{REFERENCE.start}-{REFERENCE.stop - 1}; thresholds = calibration "
        "seed's healthy max after burn-in, per platform class; CUSUM "
        "reference SD pooled within class)\n"
    )
    print(
        "| alpha | contact | fault | node | onset | ramp | sev | fleet acc "
        "healthy→faulted (tail) | stat | thr | detected | round | ttd | "
        "contacts | false loc |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    outcomes: dict[tuple, list] = {}
    for d, stats in faults:
        key = family(d)
        obs = observed_mask(d)
        c = d.get("contact")
        contact = (
            f"subs {c['sub_nodes']} every {c['period']}" if c else "every round"
        )
        if key not in healthy:
            print(
                f"| {d['config']['dirichlet_alpha']} | {d['fault']['kind']} | "
                "no healthy run for these seeds |"
            )
            continue
        _, hstats = healthy[key]
        spec = d["fault"]
        bm = d["baseline_match"] or {}
        acc = (
            f"{bm['healthy_tail_acc_mean']:.4f}→{bm['faulted_tail_acc_mean']:.4f}"
            if "healthy_tail_acc_mean" in bm
            else "n/a"
        )
        classes = platform_classes(d)
        for name in STATS:
            if name not in stats or name not in hstats:
                continue
            thr = calibrate_by_class(hstats[name], classes, name, BURN_IN)
            fl = flags_by_class(stats[name], thr, classes)
            out = score_flags(
                fl, BURN_IN, spec["node"], spec["onset"], observed=obs
            )
            outcomes.setdefault(
                (
                    d["config"]["dataset"],
                    d["config"]["dirichlet_alpha"],
                    contact,
                    spec["kind"],
                    spec["ramp"],
                    spec["severity"],
                    spec["node"],
                    name,
                ),
                [],
            ).append(
                (
                    out,
                    far.get((key, name)),
                    d["config"]["rounds"] - max(spec["onset"], BURN_IN),
                )
            )
            thr_txt = " / ".join(
                f"{c}:{t.value:.3f} (n={t.n_calibration})"
                for c, t in thr.items()
            )
            print(
                f"| {d['config']['dirichlet_alpha']} | {contact} | "
                f"{spec['kind']} | {spec['node']} | {spec['onset']} | "
                f"{spec['ramp']} | {spec['severity']} | {acc} | {name} | "
                f"{thr_txt} | "
                f"{'YES' if out.detected else 'no'} | "
                f"{out.detection_round if out.detected else '-'} | "
                f"{out.time_to_detection if out.detected else '-'} | "
                f"{out.contacts_to_detection if out.detected else '-'} | "
                f"{out.false_localisations} |"
            )
    # ── detection probability over seeds ──
    print(
        "\n## Detection probability over train seeds (thresholds from seed 0; "
        "Wilson 95% interval)\n"
    )
    print(
        "Pd by chance = probability the held-out node-round FAR alone flags "
        "the target somewhere in the post-onset window.\n"
    )
    print(
        "| dataset | alpha | contact | fault | ramp | sev | node | stat | "
        "seeds | Pd | 95% CI | Pd by chance | ttd (median over hits) | false "
        "loc per run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(
        outcomes,
        key=lambda k: (
            k[0],
            k[1],
            k[2],
            k[3],
            k[4],
            k[5],
            k[6],
            STATS.index(k[7]),
        ),
    ):
        rows = outcomes[key]
        outs = [r[0] for r in rows]
        n = len(outs)
        hits = [o for o in outs if o.detected]
        pd = len(hits) / n
        lo, hi = wilson(len(hits), n)
        ttd = np.median([o.time_to_detection for o in hits]) if hits else None
        fl = np.mean([o.false_localisations for o in outs])
        f0, window = rows[0][1], rows[0][2]
        chance = "n/a" if f0 is None else f"{1 - (1 - f0) ** window:.2f}"
        dataset, alpha, contact, kind, ramp, sev, node, name = key
        summary["pd"].append(
            {
                "dataset": dataset,
                "alpha": alpha,
                "contact": contact,
                "kind": kind,
                "ramp": ramp,
                "severity": sev,
                "node": node,
                "stat": name,
                "seeds": n,
                "pd": pd,
                "lo": lo,
                "hi": hi,
                "chance": None if f0 is None else 1 - (1 - f0) ** window,
                "ttd_median": None if ttd is None else float(ttd),
                "false_loc": float(fl),
            }
        )
        print(
            f"| {dataset} | {alpha} | {contact} | {kind} | {ramp} | {sev} | "
            f"{node} | {name} | {n} | {pd:.2f} | "
            f"[{lo:.2f}, {hi:.2f}] | {chance} | "
            f"{'-' if ttd is None else f'{ttd:.0f}'} | {fl:.1f} |"
        )
    # ── threshold sweep: FAR vs Pd per alpha per statistic ──
    print(
        "\n## Threshold sweep (every-round families; Pd over the top severity "
        "of each fault kind, all seeds)\n"
    )
    print(
        "| dataset | alpha | stat | threshold | held-out node-round FAR | "
        "any-node alarm rate | Pd (n runs) | ttd median | false loc per run |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    # top severity per (alpha, kind) among severities that ran on every seed,
    # so each kind contributes the same number of runs
    counts: dict[tuple, int] = {}
    for d, _ in faults:
        if d.get("contact") or d["fault"]["ramp"] != 10:
            continue
        k = (
            d["config"]["dataset"],
            d["config"]["dirichlet_alpha"],
            d["fault"]["kind"],
            d["fault"]["severity"],
        )
        counts[k] = counts.get(k, 0) + 1
    full = max(counts.values()) if counts else 0
    top = {}
    for (ds_, alpha_, kind_, sev_), n_ in counts.items():
        if n_ == full:
            top[(ds_, alpha_, kind_)] = max(
                top.get((ds_, alpha_, kind_), 0.0), sev_
            )
    for fam, runs in sorted(
        families.items(),
        key=lambda kv: next(iter(kv[1].values()))[0]["config"][
            "dirichlet_alpha"
        ],
    ):
        seeds = sorted(runs)
        calib_d, calib_stats = runs[seeds[0]]
        if calib_d.get("contact") or len(seeds) < 2:
            continue
        alpha = calib_d["config"]["dirichlet_alpha"]
        classes = platform_classes(calib_d)
        fam_faults = [
            (d, st)
            for d, st in faults
            if family(d) == fam
            and d["fault"]["ramp"] == 10
            and d["fault"]["severity"]
            == top.get(
                (calib_d["config"]["dataset"], alpha, d["fault"]["kind"])
            )
        ]
        for name in STATS:
            if (
                name not in calib_stats
                or any(name not in runs[sd][1] for sd in seeds[1:])
                or any(name not in st for _, st in fam_faults)
            ):
                continue
            for q in QUANTILES:
                thr = calibrate_by_class(
                    calib_stats[name], classes, name, BURN_IN, q
                )
                fars, anys = [], []
                for sd in seeds[1:]:
                    series = runs[sd][1][name]
                    fl = flags_by_class(series, thr, classes)
                    fars.append(node_round_rate(fl, series, BURN_IN))
                    anys.append(round_alarm_rate(fl, series, BURN_IN))
                outs = [
                    score_flags(
                        flags_by_class(st[name], thr, classes),
                        BURN_IN,
                        d["fault"]["node"],
                        d["fault"]["onset"],
                    )
                    for d, st in fam_faults
                ]
                hits = [o for o in outs if o.detected]
                pd = len(hits) / len(outs) if outs else float("nan")
                ttd = (
                    float(np.median([o.time_to_detection for o in hits]))
                    if hits
                    else None
                )
                fl_mean = (
                    float(np.mean([o.false_localisations for o in outs]))
                    if outs
                    else float("nan")
                )
                label = "max" if q is None else f"q{q}"
                summary["sweep"].append(
                    {
                        "dataset": calib_d["config"]["dataset"],
                        "alpha": alpha,
                        "stat": name,
                        "threshold": label,
                        "node_round_far": float(np.mean(fars)),
                        "any_alarm": float(np.mean(anys)),
                        "pd": pd,
                        "n": len(outs),
                        "ttd_median": ttd,
                        "false_loc": fl_mean,
                    }
                )
                print(
                    f"| {calib_d['config']['dataset']} | {alpha} | {name} | "
                    f"{label} | {np.mean(fars):.3f} | {np.mean(anys):.3f} | "
                    f"{pd:.2f} ({len(outs)}) | "
                    f"{'-' if ttd is None else f'{ttd:.0f}'} | {fl_mean:.1f} |"
                )

    native_fedrr(families, faults, summary)
    (RESULTS / "scores.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote results/scores.json")
    return 0


def native_fedrr(
    families: dict, faults: list[tuple[dict, dict]], summary: dict
) -> None:
    """Score FedRR at its own control limit, as its paper would run it.

    H is the limit that gives an in-control ARL of FEDRR_ARL0 under random
    ranks. That guarantee assumes exchangeable (IID) clients; the false-alarm
    rate on held-out healthy seeds shows what it is worth under heterogeneity.
    Every seed is held out here, since nothing is calibrated on the runs.
    """
    every_round = [
        (fam, runs)
        for fam, runs in families.items()
        if not next(iter(runs.values()))[0].get("contact")
        and all("fedrr_steps" in st for _, st in runs.values())
    ]
    if not every_round:
        return
    k = next(iter(every_round[0][1].values()))[0]["config"]["n_clients"]
    h = fedrr_threshold(FEDRR_ARL0, k, FEDRR_D)
    print(
        f"\n## FedRR at its own control limit (ARL0 {FEDRR_ARL0}, d "
        f"{FEDRR_D}, K {k}: H = {h:.2f}); every seed held out\n"
    )
    print(
        "| dataset | alpha | stat | healthy runs | node-round FAR | "
        "rounds with any alarm | runs with any alarm | fault runs | "
        "detected | false loc per run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    summary["fedrr_native"] = {"h": h, "arl0": FEDRR_ARL0, "rows": []}
    for fam, runs in sorted(
        every_round,
        key=lambda kv: next(iter(kv[1].values()))[0]["config"][
            "dirichlet_alpha"
        ],
    ):
        d0 = next(iter(runs.values()))[0]
        classes = platform_classes(d0)
        for name in ("fedrr", "fedrr_norm", "fedrr_steps", "fedrr_norm_steps"):
            thr = {
                c: Threshold(name, h, FEDRR_START, f"arl0:{FEDRR_ARL0}")
                for c in np.unique(classes)
            }
            fl = [
                flags_by_class(st[name], thr, classes)[FEDRR_START:]
                for _, st in runs.values()
            ]
            outs = [
                score_flags(
                    flags_by_class(st[name], thr, classes),
                    FEDRR_START,
                    d["fault"]["node"],
                    d["fault"]["onset"],
                )
                for d, st in faults
                if family(d) == fam and name in st
            ]
            row = {
                "dataset": d0["config"]["dataset"],
                "alpha": d0["config"]["dirichlet_alpha"],
                "stat": name,
                "healthy_runs": len(fl),
                "node_round_far": float(np.mean([f.mean() for f in fl])),
                "any_alarm": float(np.mean([f.any(axis=1).mean() for f in fl])),
                "runs_any_alarm": float(np.mean([f.any() for f in fl])),
                "fault_runs": len(outs),
                "detected": sum(o.detected for o in outs),
                "false_loc": float(
                    np.mean([o.false_localisations for o in outs])
                )
                if outs
                else float("nan"),
            }
            summary["fedrr_native"]["rows"].append(row)
            print(
                f"| {row['dataset']} | {row['alpha']} | {name} | "
                f"{row['healthy_runs']} | {row['node_round_far']:.3f} | "
                f"{row['any_alarm']:.3f} | {row['runs_any_alarm']:.2f} | "
                f"{row['fault_runs']} | {row['detected']} | "
                f"{row['false_loc']:.1f} |"
            )


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


if __name__ == "__main__":
    raise SystemExit(main())
