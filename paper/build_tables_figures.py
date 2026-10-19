#!/usr/bin/env python3
"""Build the paper's tables and figures from telltale's tracked result files.

Every number is read from the telltale repository's results/ directory; none
is typed in. Writes tables/*.tex and figures/*.png next to this script.

    python3 build_tables_figures.py /path/to/telltale

Population conventions (stated in the paper, Section 5):
  pooled   mid severity of each fault kind on clients 2, 5 and 9, the
           population of the threshold sweep (48 runs per alpha)
  client 2 every severity on client 2, six seeds, ten-round ramp
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
TABLES, FIGURES = HERE / "tables", HERE / "figures"
ALPHAS = (0.1, 0.5, 100.0)
ALPHA_TEX = {0.1: "0.1", 0.5: "0.5", 100.0: "100"}
ALPHA_LABEL = {
    0.1: "$\\alpha = 0.1$ (severe skew)",
    0.5: "$\\alpha = 0.5$ (moderate skew)",
    100.0: "$\\alpha = 100$ (near-IID)",
}
TOP = {"label_noise": 0.5, "blur": 1.5, "gain": 1.0, "bias": 0.5}
MID = {"label_noise": 0.3, "blur": 1.0, "gain": 0.5, "bias": 0.5}
SEVERITIES = {
    "label_noise": (0.1, 0.3, 0.5),
    "blur": (0.5, 1.0, 1.5),
    "gain": (0.25, 0.5, 1.0),
    "bias": (0.5,),
}
KIND_TEX = {
    "label_noise": "label noise",
    "blur": "blur (px)",
    "gain": "gain",
    "bias": "bias (SD)",
}
COLOR = {100.0: "#2a78d6", 0.5: "#eb6834", 0.1: "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k of n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def wilson_key(n: int) -> str:
    """'0: [0.00, 0.39]; 1: ...' for every k of n."""
    return "; ".join(
        f"{k}: [{wilson(k, n)[0]:.2f}, {wilson(k, n)[1]:.2f}]"
        for k in range(n + 1)
    )


def pd_rows(scores: dict, alpha: float, stat: str, node: int) -> list[dict]:
    """Detection records for one alpha, statistic and faulted client."""
    return [
        r
        for r in scores["pd"]
        if r["dataset"] == "fashion_mnist"
        and r["alpha"] == alpha
        and r["stat"] == stat
        and r["node"] == node
        and r.get("contact") == "every round"
        and r.get("ramp") == 10
    ]


def cell(
    scores: dict, alpha: float, stat: str, kind: str, sev: float, node: int = 2
) -> dict | None:
    """The one detection record for a fault kind and severity, or None."""
    rs = [
        r
        for r in pd_rows(scores, alpha, stat, node)
        if r["kind"] == kind and abs(r["severity"] - sev) < 1e-9
    ]
    return rs[0] if rs else None


def k_of_n(r: dict | None) -> str:
    """'k/n' from a detection record."""
    if r is None:
        return "--"
    return f"{round(r['pd'] * r['seeds'])}/{r['seeds']}"


def ttd(r: dict | None) -> str:
    """Median time to detection, or a dash."""
    if r is None or r["ttd_median"] is None:
        return "--"
    v = r["ttd_median"]
    return f"{v:g}"


def sweep_max(scores: dict, alpha: float, stat: str) -> dict | None:
    """The healthy-maximum threshold row of the sweep."""
    rs = [
        r
        for r in scores["sweep"]
        if r["dataset"] == "fashion_mnist"
        and r["alpha"] == alpha
        and r["stat"] == stat
        and r["threshold"] == "max"
    ]
    return rs[0] if rs else None


def far_row(scores: dict, alpha: float, stat: str) -> dict | None:
    """Held-out false-alarm record for a statistic, every-round contact."""
    rs = [
        r
        for r in scores["far"]
        if r["dataset"] == "fashion_mnist"
        and r["alpha"] == alpha
        and r["stat"] == stat
        and not r["contact"]
    ]
    return rs[0] if rs else None


def table_kinds(scores: dict) -> str:
    """Table 1: detection by fault kind at the top severity, client 2."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{$\\alpha = {ALPHA_TEX[a]}$}}"
            for a in ALPHAS
        )
        + " \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Fault (top severity) & "
        + " & ".join(["CUSUM", "TTD", "Persist."] * 3)
        + " \\\\",
        "\\midrule",
    ]
    for kind, sev in TOP.items():
        cells = []
        for a in ALPHAS:
            c = cell(scores, a, "level_cusum", kind, sev)
            p = cell(scores, a, "persistence", kind, sev)
            cells += [k_of_n(c), ttd(c), k_of_n(p)]
        lines.append(
            f"{KIND_TEX[kind]} {sev:g} & " + " & ".join(cells) + " \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Detection of the highest severity of each fault kind on "
        "client 2, six seeds, thresholds at the healthy maximum of seed 0. "
        "CUSUM and persistence: seeds detected; TTD: median rounds from "
        "onset to the first CUSUM flag. The one bias detection at $\\alpha = 0.1$ flagged in the "
        "onset round (TTD 0: flagged in the onset round itself) and is more "
        "likely a chance alarm than a detection. Wilson 95\\% intervals for "
        "$k$ of 6: " + wilson_key(6) + ". The full grid is in "
        "Appendix~\\ref{app:grid}.}",
        "\\label{tab:kinds}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


DETECTORS = [
    ("level_cusum", "Probe CUSUM (the method)"),
    ("level_z", "Probe within-round $z$"),
    ("update_norm_cusum", "Update norm, same CUSUM"),
    ("update_loo_cusum", "LOO update distance, same CUSUM"),
    ("fedrr", "FedRR, healthy-max threshold"),
]


def fmt_pct(x: float | None, digits: int = 1) -> str:
    """Percentage or a dash."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{100 * x:.{digits}f}"


def table_detectors(scores: dict) -> str:
    """Table 2: every detector on the pooled population, per alpha."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{$\\alpha = {ALPHA_TEX[a]}$}}"
            for a in ALPHAS
        )
        + " \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Detector & " + " & ".join(["FAR \\%", "Pd", "TTD"] * 3) + " \\\\",
        "\\midrule",
    ]
    for stat, name in DETECTORS:
        cells = []
        for a in ALPHAS:
            s = sweep_max(scores, a, stat)
            if s is None or not s["n"] or s["n"] < 48:
                cells += ["--", "--", "--"]
                continue
            cells += [
                fmt_pct(s["node_round_far"]),
                f"{s['pd']:.2f}",
                "--" if s["ttd_median"] is None else f"{s['ttd_median']:g}",
            ]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Every detector with seed 0 as the calibration seed (LOO: leave-one-out). The pooled "
        "population (mid severity of each kind on clients 2, 5 and 9, 48 "
        "fault runs per level), threshold at each statistic's healthy "
        "maximum on seed 0, false alarms per client per round on held-out seeds 1 "
        "to 5, Pd the fraction detected, TTD the median rounds to the first "
        "flag. The update-space statistics cannot be scored on the full "
        "population at $\\alpha = 0.1$ (no update records for clients 5 and "
        "9 there). Medians and ranges over every choice of calibration seed "
        "are in Table~\\ref{tab:rotation}; FedRR at its own control limit is "
        "in Table~\\ref{tab:fedrr}.}",
        "\\label{tab:detectors}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def count_mid(scores: dict, alpha: float, stat: str, node: int) -> str:
    """'k/n' over the mid severity of every fault kind on one client."""
    rs = [
        r
        for r in pd_rows(scores, alpha, stat, node)
        if r["kind"] in MID and abs(r["severity"] - MID[r["kind"]]) < 1e-9
    ]
    if not rs:
        return "--"
    k = round(sum(r["pd"] * r["seeds"] for r in rs))
    n = sum(r["seeds"] for r in rs)
    return f"{k}/{n}"


def table_clients(scores: dict) -> str:
    """Table 3: detection by faulted client, mid severity."""
    stats = [
        ("level_cusum", "Probe CUSUM"),
        ("level_z", "Probe within-round $z$"),
        ("update_norm_cusum", "Update norm, same CUSUM"),
        ("fedrr", "FedRR, healthy-max threshold"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{$\\alpha = {ALPHA_TEX[a]}$}}"
            for a in ALPHAS
        )
        + " \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Detector & " + " & ".join(["c2", "c5", "c9"] * 3) + " \\\\",
        "\\midrule",
    ]
    for stat, name in stats:
        cells = [
            count_mid(scores, a, stat, n) for a in ALPHAS for n in (2, 5, 9)
        ]
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Fault runs detected by which client is faulted, mid "
        "severity of each kind (client 2: six seeds, 24 runs; clients 5 and "
        "9: three seeds, 12 runs). Clients 5 and 9 at $\\alpha = 0.1$ have "
        "no update records, so the update-space detectors are scored on "
        "client 2 only there.}",
        "\\label{tab:clients}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table_grid(scores: dict) -> str:
    """Appendix table: every severity, three probe statistics, client 2."""
    stats = [
        ("level_cusum", "CUSUM"),
        ("persistence", "Persist."),
        ("level_z", "$z$"),
    ]
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{l" + "ccc" * 3 + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{$\\alpha = {ALPHA_TEX[a]}$}}"
            for a in ALPHAS
        )
        + " \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "Fault & " + " & ".join([s[1] for s in stats] * 3) + " \\\\",
        "\\midrule",
    ]
    for kind, sevs in SEVERITIES.items():
        for sev in sevs:
            cells = [
                k_of_n(cell(scores, a, st, kind, sev))
                for a in ALPHAS
                for st, _ in stats
            ]
            lines.append(
                f"{KIND_TEX[kind]} {sev:g} & " + " & ".join(cells) + " \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Seeds detected out of six, every fault on client 2, "
        "thresholds at the healthy maximum of seed 0. Wilson 95\\% "
        "intervals for $k$ of 6: " + wilson_key(6) + ".}",
        "\\label{tab:grid}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def style(ax: plt.Axes) -> None:
    """Quiet axes: no top or right spine, light grid."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def fig_operating_curve(scores: dict) -> None:
    """Pd against held-out false-alarm rate across the threshold sweep."""
    stats = [
        ("level_cusum", "probe CUSUM", "s"),
        ("update_norm_cusum", "update norm, same CUSUM", "o"),
        ("fedrr", "FedRR", "^"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
    shades = {
        "level_cusum": INK,
        "update_norm_cusum": "#eb6834",
        "fedrr": "#2a78d6",
    }
    for ax, a in zip(axes, ALPHAS, strict=True):
        for st, name, marker in stats:
            pts = sorted(
                {
                    (r["node_round_far"], r["pd"])
                    for r in scores["sweep"]
                    if r["dataset"] == "fashion_mnist"
                    and r["alpha"] == a
                    and r["stat"] == st
                    and r["n"]
                    and r["n"] >= 48
                }
            )  # a set: the maximum and a high quantile can coincide
            if not pts:
                continue
            ax.plot(
                [100 * p[0] for p in pts],
                [p[1] for p in pts],
                "-" + marker,
                color=shades[st],
                ms=5,
                lw=1.4,
                label=name,
            )
        ax.set_title(
            ALPHA_LABEL[a].replace("$", "").replace("\\alpha", "alpha"),
            loc="left",
            fontsize=9.5,
            color=INK,
        )
        ax.set_xlabel("held-out false alarms, % of client-rounds")
        ax.set_ylim(-0.03, 1.05)
        style(ax)
    axes[0].set_ylabel("Pd, pooled mid-severity faults")
    axes[1].legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "operating_curve.png", dpi=200)
    plt.close(fig)


def fig_drift_trace(repo: Path) -> None:
    """Fleet accuracy and per-client CUSUM for one slow bias drift.

    alpha 0.5, seed 0, bias ramping to 0.5 SD on client 2 from round 15,
    healthy run of the same seed on the same machine. CUSUM values are the
    exported statistics the scorer uses.
    """
    res = repo / "results"
    healthy = fault = None
    for f in sorted(res.glob("probes_baseline_*.json")):
        d = json.loads(f.read_text())
        if (
            d["config"]["dirichlet_alpha"] == 0.5
            and d["config"]["train_seed"] == 0
            and not d.get("contact")
            and not d.get("fault")
        ):
            healthy = (f, d)
            break
    for f in sorted(res.glob("fault_baseline_*.json")):
        d = json.loads(f.read_text())
        fa = d["fault"]
        if (
            fa["kind"] == "bias"
            and fa["node"] == 2
            and fa["ramp"] == 10
            and abs(fa["severity"] - 0.5) < 1e-9
            and d["config"]["train_seed"] == 0
            and not d.get("contact")
            and healthy
            and d["fed_fingerprint"] == healthy[1]["fed_fingerprint"]
        ):
            fault = (f, d)
            break
    if healthy is None or fault is None:
        raise SystemExit(
            "drift trace: matching healthy and fault runs not found"
        )
    hs = np.load(healthy[0].with_suffix(".stats.npz"))["level_cusum"]
    fs = np.load(fault[0].with_suffix(".stats.npz"))["level_cusum"]
    thr = float(np.nanmax(hs[10:]))
    rounds = np.arange(hs.shape[0])  # 0-based, as onset 15 is
    h_acc = [r["test_acc"] for r in healthy[1]["rounds"]]
    f_acc = [r["test_acc"] for r in fault[1]["rounds"]]
    cross = [
        int(rounds[i])
        for i in range(len(rounds))
        if np.isfinite(fs[i, 2]) and fs[i, 2] > thr
    ]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.0), sharex=True)
    ax = axes[0]
    ax.plot(rounds, h_acc, color=INK2, lw=1.8, label="healthy fleet")
    ax.plot(
        rounds,
        f_acc,
        color="#eb6834",
        lw=1.8,
        label="client 2 drifting from round 15",
    )
    ax.axvspan(15, 25, color="#eb6834", alpha=0.08, lw=0)
    lo = min(min(h_acc[5:]), min(f_acc[5:]))
    ax.set_ylim(lo - 0.01, max(max(h_acc), max(f_acc)) + 0.01)
    ax.set_ylabel("fleet test accuracy")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    style(ax)
    ax = axes[1]
    for k in range(fs.shape[1]):
        if k != 2:
            ax.plot(rounds, fs[:, k], color=GRID, lw=1.1)
    ax.plot(rounds, fs[:, 2], color="#eb6834", lw=2.0, label="client 2")
    ax.plot([], [], color=GRID, lw=1.1, label="other nine clients")
    ax.axhline(thr, color=INK, lw=0.9, ls="--")
    ax.annotate(
        f"healthy maximum {thr:.2f}",
        (5, thr),
        xytext=(0, 4),
        textcoords="offset points",
        color=INK2,
        fontsize=8.5,
    )
    ax.axvspan(15, 25, color="#eb6834", alpha=0.08, lw=0)
    ax.set_xlabel("round")
    ax.set_xlim(4.5, rounds[-1] + 0.5)
    ax.set_ylabel("CUSUM of client's level")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "drift_trace.png", dpi=200)
    plt.close(fig)
    (FIGURES / "drift_trace.json").write_text(
        json.dumps(
            {
                "healthy": healthy[0].name,
                "fault": fault[0].name,
                "threshold": thr,
                "first_round_above": cross[0] if cross else None,
                "onset_round": 15,
            },
            indent=2,
        )
    )


ROTATION_NAMES = [
    ("level_cusum", "Probe CUSUM (method)"),
    ("level_z", "Probe within-round $z$"),
    ("update_norm_cusum", "Update norm CUSUM"),
    ("update_loo_cusum", "LOO update CUSUM"),
    ("fedrr", "FedRR, healthy max"),
    ("fedrr_steps", "FedRR, step-normalised"),
]


def table_rotation(repo: Path) -> str:
    """Table: medians and ranges over the six choices of calibration seed."""
    import statistics

    rows = json.loads(
        (repo / "results" / "calibration_rotation.json").read_text()
    )
    by = {}
    for r in rows:
        by.setdefault((r["stat"], r["alpha"]), []).append(r)

    def cell(stat: str, alpha: float) -> list[str]:
        rs = by.get((stat, alpha))
        if not rs:
            return ["--", "--"]
        d = [r["detected"] for r in rs]
        f = [100 * r["node_round_far"] for r in rs]
        med = statistics.median(d)
        med_s = f"{med:g}"
        return [
            f"{med_s} ({min(d)}--{max(d)})",
            f"{statistics.median(f):.1f} ({min(f):.1f}--{max(f):.1f})",
        ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "cc" * 3 + "}",
        "\\toprule",
        " & "
        + " & ".join(
            f"\\multicolumn{{2}}{{c}}{{$\\alpha = {ALPHA_TEX[a]}$}}"
            for a in ALPHAS
        )
        + " \\\\",
        "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(3)),
        "Detector & " + " & ".join(["Caught", "FAR \\%"] * 3) + " \\\\",
        "\\midrule",
    ]
    for stat, label in ROTATION_NAMES:
        cells = [c for a in ALPHAS for c in cell(stat, a)]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Every detector with each of the six healthy seeds in turn as "
        "the calibration seed, the other five held out: median over the six "
        "choices, with the range in parentheses. Caught: mid-severity fault runs "
        "detected, of 48 per level (one fault of each kind at mid severity on "
        "clients 2, 5 and 9). FAR: held-out false alarms per client per round. At "
        "$\\alpha = 0.1$ the update-space detectors are scored on client 2 "
        "only (24 runs), the one client with update records there. LOO: "
        "leave-one-out. With one healthy run setting the threshold, a held-out "
        "healthy run raised at least one alarm in 15 of 30 calibrate-one, "
        "hold-one pairs for every detector, as exchangeable runs imply "
        "(Section~\\ref{sec:protocol}).}",
        "\\label{tab:rotation}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


POOLED_NAMES = ROTATION_NAMES


def table_pooled(repo: Path) -> str:
    """Main table: thresholds from five healthy runs, the sixth held out."""
    rows = {
        (r["stat"], r["alpha"]): r
        for r in json.loads(
            (repo / "results" / "review4_scores.json").read_text()
        )
        if r["protocol"] == "pooled5"
        and r.get("partition_seed", 0) == 0
        and r.get("clip_norm") is None
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Skew & Detector & Caught & Strict & TTD & FAR \\% & "
        "Wrong clients \\\\",
        "\\midrule",
    ]
    for i, a in enumerate(ALPHAS):
        if i:
            lines.append("\\addlinespace")
        for j, (stat, label) in enumerate(POOLED_NAMES):
            r = rows.get((stat, a))
            if r is None:
                continue
            m = r["mid"]
            lines.append(
                (f"$\\alpha = {ALPHA_TEX[a]}$" if j == 0 else "")
                + f" & {label} & {m['detected']}/{m['n']} & "
                f"{m['strict']}/{m['n']} & "
                + ("--" if m["ttd_median"] is None else f"{m['ttd_median']:g}")
                + f" & {fmt_pct(r['node_round_far'])} & "
                f"{m['false_clients_mean']:.2f} \\\\"
            )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{The main result. Each healthy seed is held out in turn and "
        "the threshold is the largest healthy value after burn-in over the "
        "other five; a fault run is scored only with the threshold whose "
        "calibration runs exclude its own seed. Caught: mid-severity fault "
        "runs with the faulted client flagged after onset, of 48 per level "
        "(24 for the update-space detectors at $\\alpha = 0.1$, client 2 "
        "only). Strict: caught with no healthy client flagged at or before "
        "the detection round. TTD: median rounds from onset to the first "
        "flag on the faulted client. FAR: held-out false alarms per client "
        "per round. Wrong clients: mean number of healthy clients flagged at "
        "least once per fault run. Under this protocol exactly one of the "
        "six held-out healthy runs raised an alarm for every detector at "
        "every level, the 1 in 6 that exchangeable runs imply "
        "(Section~\\ref{sec:protocol}).}",
        "\\label{tab:pooled}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


ROBUST_STATS = [
    ("level_cusum", "Probes"),
    ("update_norm_cusum", "Update norm"),
    ("update_preclip_cusum", "Pre-clip norm"),
    ("update_loo_cusum", "LOO update"),
    ("fedrr", "FedRR"),
]


def table_robust(repo: Path) -> str:
    """Partition, onset and clipping cells: client 2, mid severity, alpha 0.5."""
    res = repo / "results"
    cells = [
        ("review4_scores.json", 0, None, "Main grid"),
        ("review4_scores.json", 1, None, "Partition 1"),
        ("review4_scores.json", 2, None, "Partition 2"),
        ("review4_scores.json", 3, None, "Partition 3"),
        ("review4_scores_onset10.json", 0, None, "Onset 10"),
        ("review4_scores_onset22.json", 0, None, "Onset 22"),
        ("review4_scores.json", 0, 0.5, "Clipped at 0.5"),
    ]
    sizes = {}
    for f in sorted(res.glob("probes_*baseline_*.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        if c["dirichlet_alpha"] == 0.5 and c["dataset"] == "fashion_mnist":
            sizes[c["partition_seed"]] = d["partition_summary"]["client_sizes"][
                2
            ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lc" + "c" * len(ROBUST_STATS) + "}",
        "\\toprule",
        "Cell & Samples & " + " & ".join(n for _, n in ROBUST_STATS) + " \\\\",
        "\\midrule",
    ]
    for fname, part, clip, label in cells:
        rows = {
            r["stat"]: r
            for r in json.loads((res / fname).read_text())
            if r["protocol"] == "pooled5"
            and r["alpha"] == 0.5
            and r.get("partition_seed", 0) == part
            and r.get("clip_norm") == clip
        }
        out = []
        for stat, _ in ROBUST_STATS:
            r = rows.get(stat)
            if r is None or (stat == "update_preclip_cusum" and clip is None):
                out.append("--")
                continue
            m = r["mid_client2"]
            out.append(f"{m['detected']} ({m['strict']})")
        lines.append(
            f"{label} & {sizes.get(part, '--'):,} & "
            + " & ".join(out)
            + " \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Robustness cells at $\\alpha = 0.5$: the mid-severity "
        "fault of each kind on client 2, six seeds, 24 fault runs per cell, "
        "thresholds from five healthy runs of the same cell with the sixth "
        "held out (as in Table~\\ref{tab:pooled}). Entries: fault runs caught, "
        "and in parentheses caught under the strict rule. Samples: client 2's "
        "training set in that partition. Main grid: partition 0, onset at "
        "round 15, no clipping. Pre-clip norm: the norm of the "
        "update the client trained, before clipping, as a client could "
        "report it; it equals the update norm when nothing is clipped. "
        "Every cell's held-out false-alarm rate is at most 0.6 percent per "
        "client per round, and one held-out healthy run in six alarms by "
        "construction.}",
        "\\label{tab:robust}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def table_fedrr_native(scores: dict) -> str:
    """Table: FedRR at its own control limit, no calibration on our runs."""
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        " & & "
        + " & ".join(f"$\\alpha = {ALPHA_TEX[a]}$" for a in ALPHAS)
        + " \\\\",
        "\\midrule",
    ]
    for j, (stat, label) in enumerate(
        (("fedrr", "FedRR"), ("fedrr_steps", "step-normalised"))
    ):
        native = {
            r["alpha"]: r
            for r in scores["fedrr_native"]["rows"]
            if r["dataset"] == "fashion_mnist" and r["stat"] == stat
        }
        if j:
            lines.append("\\addlinespace")
        lines += [
            f"{label} & Healthy runs with any alarm & "
            + " & ".join(
                f"{round(native[a]['runs_any_alarm'] * native[a]['healthy_runs'])}"
                f" of {native[a]['healthy_runs']}"
                for a in ALPHAS
            )
            + " \\\\",
            " & False alarms, \\% per client per round & "
            + " & ".join(fmt_pct(native[a]["node_round_far"]) for a in ALPHAS)
            + " \\\\",
            " & Fault runs caught & "
            + " & ".join(
                f"{native[a]['detected']} of {native[a]['fault_runs']}"
                for a in ALPHAS
            )
            + " \\\\",
        ]
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{FedRR at its own control limit ($H = "
        f"{scores['fedrr_native']['h']:.2f}$ for $K = 10$, $d = 0.5$, "
        "in-control average run length 30, as in its paper). No run is used "
        "for calibration, so all six healthy runs per level count. "
        "Step-normalised: the residual divided by the client's number of "
        "local SGD steps that round, a rough correction for clients with "
        "more data taking more steps; it does not make clients exchangeable. "
        "Fault runs: every fault run FedRR could be "
        "scored on at that level, all severities.}",
        "\\label{tab:fedrr}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Write every table and figure; print what was written."""
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    repo = Path(sys.argv[1]).resolve()
    scores = json.loads((repo / "results" / "scores.json").read_text())
    TABLES.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (TABLES / "kinds.tex").write_text(table_kinds(scores))
    (TABLES / "detectors.tex").write_text(table_detectors(scores))
    (TABLES / "clients.tex").write_text(table_clients(scores))
    (TABLES / "grid.tex").write_text(table_grid(scores))
    (TABLES / "rotation.tex").write_text(table_rotation(repo))
    (TABLES / "fedrr.tex").write_text(table_fedrr_native(scores))
    (TABLES / "pooled.tex").write_text(table_pooled(repo))
    (TABLES / "robust.tex").write_text(table_robust(repo))
    fig_operating_curve(scores)
    fig_drift_trace(repo)
    print(
        "wrote",
        sorted(p.name for p in TABLES.iterdir()),
        sorted(p.name for p in FIGURES.iterdir()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
