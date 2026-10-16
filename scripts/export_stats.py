#!/usr/bin/env python3
"""Write each run's probe statistics to <stem>.stats.npz.

The probe-score array of one run is several megabytes; the statistics the
scorer derives from it are a few [round, node] series and depend on that run
alone. A machine that runs part of the grid can export these and send them
back instead of the arrays, and score_faults.py reads them when the array is
absent. Runs that already have a .stats.npz are skipped unless --force.

    python scripts/export_stats.py                 # every run with an array
    python scripts/export_stats.py --stems a b     # only these result stems
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from score_faults import platform_classes, probe_statistics  # noqa: E402

RESULTS = REPO / "results"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stems", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    stems = args.stems or [
        f.stem
        for f in sorted(RESULTS.glob("*.json"))
        if (RESULTS / f"{f.stem}.npz").exists()
    ]
    n = 0
    for stem in stems:
        out = RESULTS / f"{stem}.stats.npz"
        if out.exists() and not args.force:
            continue
        d = json.loads((RESULTS / f"{stem}.json").read_text())
        probs = np.load(RESULTS / f"{stem}.npz")["probs"]
        np.savez_compressed(out, **probe_statistics(probs, platform_classes(d)))
        n += 1
    print(f"wrote {n} .stats.npz files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
