#!/bin/bash
# Reproduce every number in the README from scratch, in order, with the price
# printed first.
# Stage 1-2 healthy arms: ~20 min. Stage 4 grid: ~10 h. Acoustic grid: ~5 h.
# Node grid: ~8 h.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m pytest -q
echo "== healthy arms (stage 1-2), ~20 min"
for spec in "0.5 baseline" "100 iid" "0.1 severe"; do
    set -- $spec
    $PY scripts/run_probes.py --alpha $1 --tag $2
done
$PY scripts/summarise_probes.py
echo "== submarine arm, ~20 min"
$PY scripts/run_probes.py --alpha 0.5 --sub-nodes 2,5,8 --contact-period 3
$PY scripts/run_probes.py --alpha 0.5 --sub-nodes 2,5,8 --contact-period 3 \
    --kind label_noise --node 2 --onset 15 --ramp 10 --severity 0.3
$PY scripts/run_probes.py --alpha 0.5 --sub-nodes 2,5,8 --contact-period 3 \
    --kind bias --node 2 --onset 15 --ramp 10 --severity 0.5
echo "== grids; each prices itself and skips what exists"
$PY scripts/run_grid.py
$PY scripts/run_grid.py --nodes
$PY scripts/run_grid.py --dataset deepship --local-epochs 5 --tag-prefix ds- \
    --minutes-per-run 3
$PY scripts/score_faults.py
$PY scripts/make_figures.py
