"""Score the batch. No run is used both to set a threshold and to be judged.

Threshold = the highest CUSUM any client reached in any round of the
calibration healthy runs. Every way of choosing 3 of the 5 healthy runs for
calibration is tried, so the result does not hang on one lucky split. The
other 2 healthy runs measure false alarms. Fault runs are judged against the
same threshold.

    python3 score.py
"""

import itertools
import json
from pathlib import Path

RES = Path(__file__).parent / "results"


def load(name):
    d = json.loads((RES / f"{name}.json").read_text())
    rows = []  # (round, site, cusum)
    for rec in d["records"]:
        for cid, v in rec["cusum"].items():
            rows.append((rec["round"], rec["site"][cid], v))
    return d, rows


def main():
    healthy = sorted(p.stem for p in RES.glob("healthy-*.json"))
    faults = sorted(p.stem for p in RES.glob("*-site*.json"))
    data = {n: load(n) for n in healthy + faults}

    print("runs:")
    for n in healthy + faults:
        d, rows = data[n]
        peak = max((v for _, _, v in rows), default=float("nan"))
        scored = {rec["n_scored"] for rec in d["records"]}
        print(
            f"  {n:18s} {d['status']:9s} rounds={len(d['records'])} clients scored per round={sorted(scored)} errors={len(d['errors'])} peak CUSUM={peak:.2f}"
        )

    print(
        "\nper fault run, CUSUM of the faulty site and the highest other site (final round):"
    )
    for n in faults:
        d, rows = data[n]
        f = d["fault"]
        last = max(r for r, _, _ in rows)
        bad = [v for r, s, v in rows if r == last and s == f["site"]][0]
        other = max(v for r, s, v in rows if r == last and s != f["site"])
        print(
            f"  {n:18s} faulty site {f['site']}: {bad:8.2f}   best other: {other:6.2f}"
        )

    print(
        "\nrotation over calibration sets (3 healthy runs set the threshold):"
    )
    caught = {n: 0 for n in faults}
    wrong = {n: 0 for n in faults}
    delays = {n: [] for n in faults}
    fa_rounds = fa_total = fa_runs = heldout_runs = 0
    rotations = list(itertools.combinations(healthy, 3))
    for cal in rotations:
        thr = max(v for n in cal for _, _, v in data[n][1])
        for n in healthy:
            if n in cal:
                continue
            rows = data[n][1]
            over = sum(1 for _, _, v in rows if v > thr)
            fa_rounds += over
            fa_total += len(rows)
            fa_runs += 1 if over else 0
            heldout_runs += 1
        for n in faults:
            d, rows = data[n]
            f = d["fault"]
            hits = sorted(
                r
                for r, s, v in rows
                if s == f["site"] and v > thr and r >= f["onset"]
            )
            early = [
                r
                for r, s, v in rows
                if s == f["site"] and v > thr and r < f["onset"]
            ]
            others = [(r, s) for r, s, v in rows if s != f["site"] and v > thr]
            if hits and not early:
                caught[n] += 1
                delays[n].append(hits[0] - f["onset"])
            if others or early:
                wrong[n] += 1
    R = len(rotations)
    print(f"  rotations: {R}")
    print(
        f"  held-out healthy: {fa_rounds} of {fa_total} client-rounds above threshold; {fa_runs} of {heldout_runs} held-out runs had any alarm"
    )
    for n in faults:
        dl = delays[n]
        rng = f"{min(dl)} to {max(dl)} rounds after onset" if dl else "n/a"
        print(
            f"  {n:18s} right site flagged in {caught[n]} of {R} rotations ({rng}); a wrong site or an early flag in {wrong[n]} of {R}"
        )

    # Second rule, written 2026-11-16 after seeing labelnoise-site3 and BEFORE
    # any blur result was read. labelnoise-site3 is therefore not a fair test
    # of it; the blur runs are. A client is flagged only if its CUSUM is above
    # the healthy threshold AND it is an outlier among this round's CUSUMs:
    # robust z (median and scaled MAD across clients) above 3.5, the standard
    # Iglewicz-Hoaglin cutoff. Not tuned on any run.
    def outliers(vals):
        xs = sorted(vals.values())
        n = len(xs)
        med = (xs[(n - 1) // 2] + xs[n // 2]) / 2
        dev = sorted(abs(v - med) for v in xs)
        mad = (dev[(n - 1) // 2] + dev[n // 2]) / 2 * 1.4826
        if mad <= 0:
            return {s for s, v in vals.items() if v > med}
        return {s for s, v in vals.items() if (v - med) / mad > 3.5}

    print("\nrule 2 (above threshold AND an outlier among clients that round):")
    caught2 = {n: 0 for n in faults}
    wrong2 = {n: 0 for n in faults}
    delays2 = {n: [] for n in faults}
    fa2 = 0
    for cal in rotations:
        thr = max(v for n in cal for _, _, v in data[n][1])
        for n in healthy + faults:
            if n in cal:
                continue
            by_round = {}
            for r, s, v in data[n][1]:
                by_round.setdefault(r, {})[s] = v
            flags = sorted(
                (r, s)
                for r, vals in by_round.items()
                for s in outliers(vals)
                if vals[s] > thr
            )
            if n in healthy:
                fa2 += 1 if flags else 0
                continue
            f = data[n][0]["fault"]
            hits = [r for r, s in flags if s == f["site"] and r >= f["onset"]]
            bad = [(r, s) for r, s in flags if s != f["site"] or r < f["onset"]]
            if hits and not bad:
                caught2[n] += 1
                delays2[n].append(hits[0] - f["onset"])
            if bad:
                wrong2[n] += 1
    print(f"  held-out healthy runs with any alarm: {fa2} of {heldout_runs}")
    for n in faults:
        dl = delays2[n]
        rng = f"{min(dl)} to {max(dl)} rounds after onset" if dl else "n/a"
        print(
            f"  {n:18s} right site only in {caught2[n]} of {R} ({rng}); a wrong site or an early flag in {wrong2[n]} of {R}"
        )

    out = {
        "rule2": {
            "heldout_runs_with_alarm": [fa2, heldout_runs],
            "faults": {
                n: {
                    "caught": caught2[n],
                    "wrong_or_early": wrong2[n],
                    "delays": delays2[n],
                }
                for n in faults
            },
        },
        "rotations": R,
        "heldout_false_alarm_client_rounds": [fa_rounds, fa_total],
        "heldout_runs_with_alarm": [fa_runs, heldout_runs],
        "faults": {
            n: {
                "caught": caught[n],
                "wrong_or_early": wrong[n],
                "delays": delays[n],
                "fault": data[n][0]["fault"],
            }
            for n in faults
        },
    }
    (RES / "SCORES.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
