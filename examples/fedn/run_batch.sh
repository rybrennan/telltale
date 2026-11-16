#!/bin/bash
# The full batch: five healthy runs, then four runs with one faulty client.
# 40 rounds each. Faults start at round 20 and ramp over 10 rounds.
cd "$(dirname "$0")"
SEED_MODEL=46c4c7e5-cdfa-4c95-bfea-182d5346610c
PY=${PY:-python}  # an environment with fedn 0.33.0 installed
for i in 1 2 3 4 5; do
  $PY run_experiment.py healthy-$i --rounds 40 --seed $i --seed-model $SEED_MODEL
done
$PY run_experiment.py labelnoise-site3 --rounds 40 --seed 11 --fault label_noise:3:20:10:0.3 --seed-model $SEED_MODEL
$PY run_experiment.py labelnoise-site6 --rounds 40 --seed 12 --fault label_noise:6:20:10:0.3 --seed-model $SEED_MODEL
$PY run_experiment.py blur-site5 --rounds 40 --seed 13 --fault blur:5:20:10:1.0 --seed-model $SEED_MODEL
$PY run_experiment.py blur-site2 --rounds 40 --seed 14 --fault blur:2:20:10:1.0 --seed-model $SEED_MODEL
echo BATCH_DONE
