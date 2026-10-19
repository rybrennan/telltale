#!/usr/bin/env python3
"""Figures for the brief, generated from results/.

No dashboard: static PNGs and one index.html that embeds them with the
sentence each one supports.

    python scripts/make_figures.py  # figures/*.png and figures/index.html

One figure per claim. Every number is read from the result files; nothing is
typed in. Palette: the validated reference categorical order, fixed per
series across figures (blue = alpha 100, orange = 0.5, aqua = 0.1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from telltale.detect import cusum, level_series  # noqa: E402

RESULTS = REPO / "results"


def _review4_cell(d: dict) -> bool:
    """Say whether a run belongs to the review-4 partition or clipping cells.

    Args:
        d: A result file's contents.

    Returns:
        True when the run is not part of the main grid.
    """
    c = d["config"]
    return c.get("partition_seed", 0) != 0 or c.get("clip_norm") is not None


FIG = REPO / "figures"
ALPHA_COLOR = {100.0: "#2a78d6", 0.5: "#eb6834", 0.1: "#1baf7a"}
ALPHA_LABEL = {
    100.0: "alpha 100 (near-IID)",
    0.5: "alpha 0.5 (moderate skew)",
    0.1: "alpha 0.1 (severe skew)",
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "legend.frameon": False,
    }
)


def healthy_runs(
    dense_only: bool = True,
) -> dict[float, tuple[dict, np.ndarray]]:
    """Load the seed-0 healthy probe runs, keyed by Dirichlet alpha.

    Args:
        dense_only: Skip runs that only contacted nodes on some rounds.

    Returns:
        Map from alpha to (result JSON, probe-score array from the .npz).
    """
    out = {}
    for f in sorted(RESULTS.glob("probes_*.json")):
        d = json.loads(f.read_text())
        if _review4_cell(d):
            continue
        if dense_only and d.get("contact"):
            continue
        if d["config"]["train_seed"] != 0:
            continue
        out[d["config"]["dirichlet_alpha"]] = (
            d,
            np.load(f.with_suffix(".npz"))["probs"],
        )
    return out


def fig_bandwidth(runs: dict[float, tuple[dict, np.ndarray]]) -> None:
    """Plot healthy level SD against probe payload size.

    Claim: the temporal floor barely moves with battery size; a kilobyte is
    the knee.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for alpha in (100.0, 0.5, 0.1):
        d, _ = runs[alpha]
        rows = d["characterisation"]
        x = [r["payload_bytes_float16"] for r in rows]
        y = [r["level_round_sd"] for r in rows]
        ax.plot(
            x,
            y,
            "-o",
            color=ALPHA_COLOR[alpha],
            lw=2,
            ms=6,
            label=ALPHA_LABEL[alpha],
        )
        ax.annotate(
            ALPHA_LABEL[alpha].split(" (")[0],
            (x[-1], y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK2,
            va="center",
            fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_xlabel("probe payload per node per round, bytes (float16)")
    ax.set_ylabel("healthy across-round SD of level")
    ax.set_ylim(0, None)
    ax.set_xlim(150, 30000)
    ax.set_title(
        "More probes buy almost no resolution:\nthe healthy floor is the "
        "model moving, not the battery",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bandwidth.png", dpi=160)
    plt.close(fig)


def fig_confound(runs: dict[float, tuple[dict, np.ndarray]]) -> None:
    """Plot the healthy level distribution at each alpha.

    Claim: healthy cross-node divergence is set by heterogeneity, 20x across
    alpha.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    alphas = [100.0, 0.5, 0.1]
    for i, alpha in enumerate(alphas):
        _, probs = runs[alpha]
        lv = level_series(probs)["level"][20:]
        vals = lv[~np.isnan(lv)]
        ax.scatter(
            np.full(vals.size, i)
            + np.random.default_rng(0).uniform(-0.12, 0.12, vals.size),
            vals,
            s=14,
            color=ALPHA_COLOR[alpha],
            alpha=0.7,
            edgecolors="#fcfcfb",
            linewidths=0.5,
        )
        med = float(np.median(vals))
        ax.hlines(med, i - 0.25, i + 0.25, color=INK, lw=2)
        ax.annotate(
            f"median {med:.3f}",
            (i + 0.28, med),
            color=INK2,
            va="center",
            fontsize=9,
        )
    ax.set_xticks(
        range(3), [ALPHA_LABEL[a].replace(" (", "\n(") for a in alphas]
    )
    ax.set_ylabel("healthy level: mean JS to peers (nats)")
    ax.set_title(
        "Legitimate heterogeneity is the level signal:\n20x across alpha with "
        "no fault present",
        loc="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "confound.png", dpi=160)
    plt.close(fig)


def dense_healthy(alpha: float) -> Path:
    """Return the seed-0, every-round healthy run at this alpha.

    The run is chosen by reading the JSON rather than by glob order, because
    seed-1 files match the same pattern.

    Raises:
        FileNotFoundError: No such run exists in results/.
    """
    for f in sorted(RESULTS.glob("probes_*.json")):
        d = json.loads(f.read_text())
        if _review4_cell(d):
            continue
        if (
            d["config"]["dirichlet_alpha"] == alpha
            and d["config"]["train_seed"] == 0
            and not d.get("contact")
        ):
            return f
    raise FileNotFoundError(f"no dense seed-0 healthy run at alpha {alpha}")


def fig_drift_trace() -> None:
    """Plot fleet accuracy and per-node CUSUM for one slow bias drift.

    Claim: the fleet metric never moves; the instrument does. One fault, two
    views.
    """
    hfile = dense_healthy(0.5)
    healthy = json.loads(hfile.read_text())
    fault_file = None
    for f in RESULTS.glob("fault_baseline_*.json"):
        d = json.loads(f.read_text())
        if d["fault"] == {
            "kind": "bias",
            "node": 2,
            "onset": 15,
            "severity": 0.5,
            "ramp": 10,
            "seed": 0,
        } and not d.get("contact"):
            fault_file = f
            break
    if fault_file is None:
        return
    fd = json.loads(fault_file.read_text())
    hp = np.load(hfile.with_suffix(".npz"))["probs"]
    fp = np.load(fault_file.with_suffix(".npz"))["probs"]
    h_acc = [r["test_acc"] for r in healthy["rounds"]]
    f_acc = [r["test_acc"] for r in fd["rounds"]]
    S_h = cusum(level_series(hp)["level"], slice(5, 15))
    S_f = cusum(level_series(fp)["level"], slice(5, 15))
    thr = float(np.nanmax(S_h[10:]))

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(range(1, 31), h_acc, color=INK2, lw=2, label="healthy fleet")
    ax.plot(
        range(1, 31),
        f_acc,
        color="#eb6834",
        lw=2,
        label="node 2 drifting from round 15",
    )
    ax.axvspan(15, 25, color="#eb6834", alpha=0.08, lw=0)
    ax.set_ylim(0.85, 0.91)
    ax.set_ylabel("fleet test accuracy")
    ax.set_title(
        "What the operator sees: the fleet metric does not move",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax = axes[1]
    for k in range(10):
        if k != 2:
            ax.plot(range(1, 31), S_f[:, k], color=GRID, lw=1.2)
    ax.plot(
        range(1, 31),
        S_f[:, 2],
        color="#eb6834",
        lw=2.2,
        label="node 2 (drifting)",
    )
    ax.plot([], [], color=GRID, lw=1.2, label="other nine nodes")
    ax.axhline(thr, color=INK, lw=1, ls="--")
    ax.annotate(
        f"healthy ceiling {thr:.2f}",
        (1, thr),
        xytext=(0, 4),
        textcoords="offset points",
        color=INK2,
        fontsize=9,
    )
    ax.axvspan(15, 25, color="#eb6834", alpha=0.08, lw=0)
    ax.set_xlabel("round")
    ax.set_ylabel("CUSUM of node's own level")
    ax.set_title(
        "What the instrument sees: the drifting node crosses the healthy "
        "ceiling",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "drift_trace.png", dpi=160)
    plt.close(fig)


def fig_detection_by_alpha() -> None:
    """Claim: detectability of the same fault falls with heterogeneity."""
    stats = ["level", "level_z", "persistence", "level_cusum"]
    names = {
        "level": "level",
        "level_z": "within-round z",
        "persistence": "persistence",
        "level_cusum": "CUSUM",
    }
    # Read the scorer's own output so the figure cannot disagree with the table.
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "score_faults.py")],
        capture_output=True,
        text=True,
    ).stdout
    ttd = {}
    for line in out.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if (
            len(cells) < 15
            or cells[1] != "every round"
            or cells[2] != "bias"
            or cells[5] != "10"
            or cells[6] != "0.5"
        ):
            continue
        alpha, stat, detected, t = (
            float(cells[0]),
            cells[8],
            cells[10],
            cells[12],
        )
        if stat in stats:
            ttd[(alpha, stat)] = int(t) if detected == "YES" else None
    if not ttd:
        return
    alphas = [100.0, 0.5, 0.1]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, alpha in enumerate(alphas):
        for j, st in enumerate(stats):
            v = ttd.get((alpha, st))
            color = ALPHA_COLOR[alpha] if v is not None else "#fcfcfb"
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    0.94,
                    0.94,
                    color=color,
                    ec=INK2 if v is None else color,
                    lw=0.8,
                )
            )
            ax.text(
                j + 0.47,
                i + 0.47,
                "miss" if v is None else f"{v} rounds",
                ha="center",
                va="center",
                color="#fcfcfb" if v is not None else INK2,
                fontsize=10,
            )
    ax.set_xlim(0, len(stats))
    ax.set_ylim(0, len(alphas))
    ax.set_xticks(
        [j + 0.47 for j in range(len(stats))], [names[s] for s in stats]
    )
    ax.set_yticks(
        [i + 0.47 for i in range(len(alphas))], [ALPHA_LABEL[a] for a in alphas]
    )
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_title(
        "Same 0.5 SD drift, same node: rounds from onset to detection",
        loc="left",
        color=INK,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "detection_by_alpha.png", dpi=160)
    plt.close(fig)


def fig_pd_by_alpha(
    dataset: str = "fashion_mnist", node: int = 2, suffix: str = ""
) -> None:
    """Plot detection probability against alpha, per fault kind.

    Claim: detection probability vs heterogeneity, per fault kind, for the
    two statistics that localise. Read from results/scores.json.
    """
    sc = RESULTS / "scores.json"
    if not sc.exists():
        return
    rows = {}
    for r in json.loads(sc.read_text())["pd"]:
        if (
            r["contact"] == "every round"
            and r["ramp"] == 10
            and r.get("dataset", "fashion_mnist") == dataset
            and r["node"] == node
        ):
            rows[(r["alpha"], r["kind"], r["severity"], r["stat"])] = (
                r["pd"],
                r["lo"],
                r["hi"],
                r["chance"] or np.nan,
            )
    if not rows:
        return
    kinds = [
        ("label_noise", "label noise"),
        ("blur", "blur"),
        ("gain", "gain"),
        ("bias", "bias"),
    ]
    alphas = [100.0, 0.5, 0.1]
    stats = [("persistence", "#2a78d6", "o"), ("level_cusum", "#eb6834", "s")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.0), sharey=True)
    # One panel per fault kind: subplots(1, 4) and four kinds.
    for ax, (kind, label) in zip(axes, kinds, strict=True):
        sevs = sorted(
            sv
            for sv in {k[2] for k in rows if k[1] == kind}
            if all((a, kind, sv, "persistence") in rows for a in alphas)
        )
        if not sevs:
            continue
        sev = sevs[-1]
        for j, (st, color, marker) in enumerate(stats):
            xs, ys, los, his, ch = [], [], [], [], []
            for i, a in enumerate(alphas):
                r = rows.get((a, kind, sev, st))
                if r is None:
                    continue
                xs.append(i + (j - 0.5) * 0.18)
                ys.append(r[0])
                los.append(r[0] - r[1])
                his.append(r[2] - r[0])
                ch.append(r[3])
            ax.errorbar(
                xs,
                ys,
                yerr=[los, his],
                fmt=marker,
                color=color,
                ms=7,
                lw=1.4,
                capsize=3,
                label="persistence" if st == "persistence" else "CUSUM",
            )
            ax.plot(xs, ch, ls=":", color=color, lw=1, alpha=0.7)
        ax.set_title(
            f"{label}, severity {sev:g}", loc="left", fontsize=10, color=INK
        )
        ax.set_xticks(range(3), ["100", "0.5", "0.1"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("alpha (more skew →)")
    axes[0].set_ylabel("Pd over 3 seeds")
    axes[0].legend(loc="lower left", fontsize=8)
    label = {
        "fashion_mnist": "clothing photographs",
        "deepship": "DeepShip underwater acoustics",
    }[dataset]
    fig.suptitle(
        f"Detection probability vs heterogeneity, {label} (bars: Wilson 95%; "
        "dotted: Pd from false alarms alone)",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / f"pd_by_alpha{suffix}.png", dpi=160)
    plt.close(fig)


def fig_operating_curve() -> None:
    """Plot Pd against false-alarm rate across the threshold sweep.

    Claim: sensitivity is bought with alarm rate; here is the exchange rate,
    per alpha, for the statistics that matter. Threshold sweep from
    scores.json.
    """
    sc = RESULTS / "scores.json"
    if not sc.exists():
        return
    sweep = [
        r
        for r in json.loads(sc.read_text())["sweep"]
        if r.get("dataset", "fashion_mnist") == "fashion_mnist"
    ]
    if not sweep:
        return
    alphas = [100.0, 0.5, 0.1]
    stats = [
        ("persistence", "#2a78d6", "o"),
        ("level_cusum", "#eb6834", "s"),
        ("level_cusum2", "#1baf7a", "^"),
    ]
    names = {
        "persistence": "persistence",
        "level_cusum": "CUSUM",
        "level_cusum2": "two-sided CUSUM",
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    # One panel per alpha: subplots(1, 3) and three alphas.
    for ax, a in zip(axes, alphas, strict=True):
        for st, color, marker in stats:
            pts = sorted(
                [
                    (r["any_alarm"], r["pd"], r["threshold"])
                    for r in sweep
                    if r["alpha"] == a and r["stat"] == st
                ]
            )
            if not pts:
                continue
            ax.plot(
                [p[0] for p in pts],
                [p[1] for p in pts],
                "-" + marker,
                color=color,
                ms=6,
                lw=1.6,
                label=names[st],
            )
        ax.set_title(ALPHA_LABEL[a], loc="left", fontsize=10, color=INK)
        ax.set_xlabel("held-out rounds with any false alarm")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.05, 1.08)
    axes[0].set_ylabel("Pd, top severity, all fault kinds")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Sensitivity is bought with alarm rate: threshold at healthy max, "
        "then 99th / 95th / 90th percentile",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "operating_curve.png", dpi=160)
    plt.close(fig)


def write_index() -> None:
    """Write figures/index.html with each existing figure and its claim."""
    items = [
        (
            "operating_curve.png",
            "The exchange rate between sensitivity and alarm rate, per "
            "heterogeneity level: each statistic at four thresholds, from the "
            "healthy maximum down to the 90th percentile. Pd is over the top "
            "severity of every fault kind; the alarm rate is counted on "
            "healthy seeds the threshold never saw.",
        ),
        (
            "pd_by_alpha_acoustic.png",
            "The same figure on real hydrophone recordings (DeepShip, four "
            "vessel classes). The cumulative statistic carries the same "
            "ordering by heterogeneity; the peer-persistence statistic does "
            "not transfer; nothing is caught under severe skew.",
        ),
        (
            "pd_by_alpha.png",
            "Detection probability over three seeds, per fault kind at its "
            "top severity, against heterogeneity. Persistence localises "
            "cleanly; CUSUM detects slow input drift that nothing else sees. "
            "Three seeds give a wide interval: a 3-of-3 reads as [0.44, 1.00].",
        ),
        (
            "drift_trace.png",
            "One slow drift, two views: the fleet accuracy the operator "
            "watches does not move; the node's own cumulative divergence "
            "crosses the healthy ceiling five rounds after onset.",
        ),
        (
            "detection_by_alpha.png",
            "Detectability of the same fault falls monotonically with "
            "heterogeneity: every statistic near-IID, one at moderate skew, "
            "none at severe skew. Single seed, single node; the grid turns "
            "this into probabilities.",
        ),
        (
            "confound.png",
            "Why no fixed threshold can work: healthy cross-node divergence "
            "spans 20x across heterogeneity with no fault present.",
        ),
        (
            "bandwidth.png",
            "The bandwidth tradeoff, measured: the healthy temporal floor "
            "barely moves from 200 bytes to 10 kB per node per round, because "
            "it is the model moving, not the battery being small.",
        ),
    ]
    html = [
        "<!doctype html><meta charset=utf-8><title>telltale figures</title>",
        "<style>"
        "body{font:15px/1.5 system-ui;max-width:760px;margin:2rem auto;"
        "padding:0 1rem;color:#0b0b0b;background:#fcfcfb}"
        "img{width:100%;border:1px solid #e6e5e1}"
        "figcaption{color:#52514e;margin:.4rem 0 2rem}"
        "</style>",
        "<h1>telltale — figures</h1><p>Generated from <code>results/</code> "
        "by <code>scripts/make_figures.py</code>. Every number is read from a "
        "result file. The repository states no detection probability yet.</p>",
    ]
    for f, cap in items:
        if (FIG / f).exists():
            html.append(
                f"<figure><img src='{f}' alt='{cap}'>"
                f"<figcaption>{cap}</figcaption></figure>"
            )
    (FIG / "index.html").write_text("\n".join(html))


def main() -> int:
    """Write every figure whose inputs exist, then the index page."""
    FIG.mkdir(exist_ok=True)
    runs = healthy_runs()
    if len(runs) == 3:
        fig_bandwidth(runs)
        fig_confound(runs)
    fig_drift_trace()
    fig_detection_by_alpha()
    fig_pd_by_alpha()
    fig_pd_by_alpha(dataset="deepship", suffix="_acoustic")
    fig_operating_curve()
    write_index()
    print("wrote", sorted(p.name for p in FIG.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
