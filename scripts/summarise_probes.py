#!/usr/bin/env python3
"""Cross-arm summary of the healthy probe runs, for the README.

Reads every results/probes_*.json, and for the quantisation effect re-derives
the level statistic from the raw .npz reports under each payload encoding.

    python scripts/summarise_probes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from telltale.detect import level_series  # noqa: E402
from telltale.probes import quantise  # noqa: E402

RESULTS = REPO / "results"


def main() -> int:
    """Print the README's cross-arm tables for the healthy probe runs."""
    runs = []
    for f in sorted(RESULTS.glob("probes_*.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        if c.get("partition_seed", 0) != 0 or c.get("clip_norm") is not None:
            continue  # review-4 cells; scored by review4_scores.py
        d["_stem"] = f.stem
        runs.append(d)
    if not runs:
        print("no probes_*.json in results/")
        return 1
    runs.sort(key=lambda d: d["config"]["dirichlet_alpha"])

    print("## Inertness and instrument\n")
    print(
        "| alpha | final acc | baseline curve identical | train loss "
        "identical | instrument issues | minutes |"
    )
    print("|---|---|---|---|---|---|")
    for d in runs:
        bm = d["baseline_match"] or {}
        print(
            f"| {d['config']['dirichlet_alpha']} | {d['final_acc']:.4f} | "
            f"{bm.get('accuracy_curve_identical', 'n/a')} | "
            f"{bm.get('train_loss_identical', 'n/a')} | "
            f"{len(d['instrument_issues'])} | {d['elapsed_seconds'] / 60:.1f} |"
        )

    print("\n## Healthy level statistic by battery size (final ten rounds)\n")
    print(
        "| alpha | n probes | payload B (fp16) | % of model | level median | "
        "across-round SD | across-client SD | max abs z | self median | "
        "persistence median |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in runs:
        for r in d["characterisation"]:
            print(
                f"| {d['config']['dirichlet_alpha']} | {r['n_probes']} | "
                f"{r['payload_bytes_float16']:,} | "
                f"{100 * r['payload_fraction_of_model']:.2f} | "
                f"{r['level_median']:.5f} | "
                f"{r['level_round_sd']:.5f} | {r['level_client_sd']:.5f} | "
                f"{r['level_z_max']:.2f} | "
                f"{r['self_median']:.5f} | {r['persist_median']:.3f} |"
            )

    print(
        "\n## Quantisation effect on the level statistic (largest battery, "
        "final ten rounds)\n"
    )
    print(
        "| alpha | encoding | bytes | level median | max abs shift vs float32 |"
    )
    print("|---|---|---|---|---|")
    for d in runs:
        npz = RESULTS / f"{d['_stem']}.npz"
        probs = np.load(npz)["probs"]
        R = probs.shape[0]
        tail = slice(max(0, R - 10), R)
        ref = level_series(probs)["level"][tail]
        n, C = probs.shape[2], probs.shape[3]
        for enc, nbytes in (("float32", 4), ("float16", 2), ("uint8", 1)):
            lv = level_series(quantise(probs, enc))["level"][tail]
            print(
                f"| {d['config']['dirichlet_alpha']} | {enc} | "
                f"{n * C * nbytes:,} | "
                f"{np.nanmedian(lv):.5f} | {np.nanmax(np.abs(lv - ref)):.2e} |"
            )
    print(
        "\n## Is the across-round spread probe-sampling noise or training "
        "noise?\n"
    )
    print(
        "Disjoint stratified 100-probe sub-batteries of the same run. If the "
        "round-to-round\n"
        "wobble were probe-sampling noise the two series would be "
        "uncorrelated and averaging\n"
        "them would shrink the spread by 1/sqrt(2) = 0.71; if it is the model "
        "itself moving,\n"
        "they correlate and the spread does not shrink.\n"
    )
    print(
        "| alpha | median corr, disjoint sub-batteries | spread ratio, mean "
        "of two vs one |"
    )
    print("|---|---|---|")
    for d in runs:
        probs = np.load(RESULTS / f"{d['_stem']}.npz")["probs"]
        R = probs.shape[0]
        tail = slice(max(0, R - 10), R)
        if probs.shape[2] < 200:
            continue
        a = level_series(probs[:, :, :100, :])["level"][tail]
        b = level_series(probs[:, :, 100:200, :])["level"][tail]
        ca, cb = a - a.mean(axis=0), b - b.mean(axis=0)
        corr = np.median(
            [np.corrcoef(ca[:, k], cb[:, k])[0, 1] for k in range(a.shape[1])]
        )
        ratio = np.nanmean(np.nanstd((a + b) / 2, axis=0)) / np.nanmean(
            np.nanstd(a, axis=0)
        )
        print(
            f"| {d['config']['dirichlet_alpha']} | {corr:.2f} | {ratio:.2f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
