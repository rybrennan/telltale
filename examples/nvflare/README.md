# telltale with NVIDIA FLARE

Cross-round client monitoring inside an NVFlare FedAvg job. Each site trains on its own
shard, scores a shared probe battery with its post-training model, and returns the scores
(float16, about 1 kB) in `FLModel.meta`. A custom `ModelAggregator` on the server keeps
every round's scores and runs telltale's detector over them: the divergence of each site
from the leave-one-out consensus, accumulated with a CUSUM against the site's own
reference epoch. A site whose data degrades slowly, inside nominal bounds, is flagged
while the fleet accuracy shows nothing.

This follows the design discussed in [NVIDIA/NVFlare#5398](https://github.com/NVIDIA/NVFlare/issues/5398):

| Piece | Where |
| --- | --- |
| Probe reports from clients | `FLModel.meta`, via `probe_meta()` in `examples/nvflare/client.py` |
| Detector history | kept on the server by `IntegrityAggregator`, written to `--history_path` each round |
| Client exclusion | off by default; `--exclude_flagged` leaves flagged sites out of the average |
| Aggregation | sample-count-weighted average, weighted by `NUM_STEPS_CURRENT_ROUND` |

The aggregator is `telltale/integrations/nvflare.py`. Its constructor takes only JSON
types, so it exports into an NVFlare job configuration like any other component.

## Install

From the repository root, in a virtual environment:

```bash
python -m pip install -e ".[nvflare]"
```

This installs NVFlare 2.7.3 or later with PyTorch; the example was run on 2.7.3 and
checked on 2.9.0. The first run downloads Fashion-MNIST
into `data/`.

Note: NVFlare's simulator (2.7.3 and 2.9.0) launches its subprocess with an unquoted
interpreter path, so the virtual environment's path must not contain spaces.

## Run

```bash
cd examples/nvflare

# Healthy federation: 10 sites, 30 rounds, moderate label skew. Records, flags nothing.
python job.py

# Partial participation: FedAvg samples 6 of the 10 sites each round.
python job.py --participation 0.6

# site-3's images blur, ramping to 1.5 px from round 15; flag with a calibrated threshold.
python job.py --fault_site 3 --fault_kind blur --fault_severity 1.5 --threshold <healthy maximum>
```

The threshold is not a constant. Healthy divergence depends on how skewed the sites'
data are, so the threshold is the healthy maximum of the CUSUM after the reference
epoch, measured on a healthy run of the same configuration. `scripts/nvflare_eval.py`
does this calibration and the scoring below.

Each run writes the detector history (every round's scores, the CUSUM path, flags and the
sites sampled each round) to `--history_path`, default `/tmp/nvflare/telltale/history.json`.

## Results

`scripts/nvflare_eval.py`, 72 simulator runs on one laptop CPU. 10 sites, 30 rounds,
Fashion-MNIST, a 50-probe battery. Fault: site-3's images blur, ramping to 1.5 px over ten
rounds from round 15. Per cell, the threshold is the CUSUM healthy maximum after the
reference epoch (rounds 5-14) on healthy seed 0, and never sees a fault. False alarms are
counted on held-out healthy seeds 1-5, per client contact. Detection is over six separate
fault seeds, reported as "caught in k of 6 seeds" with a Wilson 95% interval, not as a
probability. Summary: `results/nvflare_eval.json`.

| Label skew (alpha) | Sites sampled per round | Threshold | False alarms, held-out contacts | Blur caught | Rounds from onset to flag |
| --- | --- | --- | --- | --- | --- |
| 100 (near-IID) | 10 of 10 | 8.76 | 1 / 750 (0.1%) | 6 / 6 [0.61, 1.00] | 6, 6, 7, 9, 6, 7 |
| 100 (near-IID) | 6 of 10 | 7.63 | 12 / 450 (2.7%) | 6 / 6 [0.61, 1.00] | 6, 10, 10, 10, 5, 10 |
| 0.5 (moderate) | 10 of 10 | 3.77 | 0 / 750 (0.0%) | 6 / 6 [0.61, 1.00] | 5, 9, 4, 5, 6, 7 |
| 0.5 (moderate) | 6 of 10 | 6.62 | 0 / 450 (0.0%) | 4 / 6 [0.30, 0.90] | 8, 6, 8, 9 |
| 0.1 (severe) | 10 of 10 | 0.71 | 5 / 750 (0.7%) | 6 / 6 [0.61, 1.00] | 6, 1, 4, 0, 5, 3 |
| 0.1 (severe) | 6 of 10 | 6.54 | 0 / 450 (0.0%) | 1 / 6 [0.03, 0.56] | 5 |

**Every site reporting.** The slow blur is caught in all six seeds at every heterogeneity
level, with at most 0.7% false alarms per held-out contact. This matches the in-process
harness (CUSUM, blur 1.5 px: caught in every seed at every alpha). Delays run zero to nine
rounds; the zero- and one-round flags are both in the severe-skew cell, whose threshold is the
lowest. At its 0.7% false-alarm rate, the chance of a false flag on that one site somewhere in
the fifteen post-onset rounds is about 10%, so those two are reported, not leaned on.

**Six of ten sites sampled per round.** Near-IID, detection holds but false alarms rise to
2.7%. Under label skew the same fault is caught in 4 of 6 seeds at alpha 0.5 and 1 of 6 at
alpha 0.1. Two things change when FedAvg samples six sites a round: each site has as few as 3
contacts in the reference epoch instead of 10, and its leave-one-out consensus is built from
a different set of peers every round. The evidence points at the second. The median
coefficient of variation of a healthy site's level over the reference epoch goes from 0.173
to 0.207 at alpha 0.5 and from 0.076 to 0.151 at alpha 0.1 (six healthy seeds); the per-cell
thresholds rise with it (3.77 to 6.62, 0.71 to 6.54), and a 1.5 px blur no longer clears them
reliably. Near-IID the variation does not rise (0.428 to 0.367).

**Two alternatives were scored on the recorded histories, under the same protocol.**

| Statistic | alpha 100, 6 of 10 | alpha 0.5, 6 of 10 | alpha 0.1, 6 of 10 | Cost elsewhere |
| --- | --- | --- | --- | --- |
| Leave-one-out consensus (shipped) | 6/6, 2.7% false alarms | 4/6, 0.0% | 1/6, 0.0% | none |
| Pooled reference spread | 6/6, 0.7% | 4/6, 0.0% | 1/6, 0.0% | none |
| Fixed peer set (consensus frozen at the reference epoch) | 6/6, 8.0% | 6/6, 0.0% | 3/6, 2.2% | near-IID full participation 5/6; severe full participation 1.2% false alarms |

Pooling the reference spread across sites, the harness's remedy for a starved reference, does
not recover detection, so the sparse reference alone is not the limit. Comparing each site
with a consensus frozen at the reference epoch recovers detection at moderate skew and half of it at severe skew,
but it was designed after the first three seeds showed the failure and it triples the
near-IID false-alarm rate under partial participation. It is not adopted; a statistic that
keeps the frozen consensus's detection without its false alarms is the next thing to measure.

Sampling in these runs is NVFlare's own: FedAvg draws a fresh random subset each round and
does not take a seed, so the 60% cells are not bitwise reproducible. The sites sampled each
round are recorded in every history file. One of the 72 simulations stalled (a site lost its
connection to the simulator's server at round 3) and was rerun.

## Validation performed

- **Code tested:** telltale `6b9e23f` (the commit the results above were scored at; later commits touch only CI and documentation).
- **NVFlare:** 2.7.3 for the 72-run evaluation; the example re-run from a fresh clone on 2.9.0 (4 sites, 4 rounds, 3 of 4 sampled per round).
- **Environment:** macOS 15.6 (Apple M4 Pro), Python 3.12, CPU only, one machine. The same 4-site run also executes in the simulator on Ubuntu in this repository's CI on every push.
- **Commands:** `python job.py` with the defaults for one healthy run (input: Fashion-MNIST, downloaded to `data/` on first run); `python scripts/nvflare_eval.py run --jobs 3` then `score` for the table above.
- **Observed result:** the table above; per-run detector histories (every round's probe scores, CUSUM path, flags, sampled sites) are written by the aggregator, and the scored summary is `results/nvflare_eval.json`. Representative log: the 4-site simulator run in this repository's CI, [latest run on main](https://github.com/rybrennan/telltale/actions/runs/38508896389), step "NVFlare example, 4 sites x 4 rounds in the simulator".
- **Limitations:** simulator only, one fault type (blur), one dataset (Fashion-MNIST), three heterogeneity levels, six seeds per cell. Detection degrades under partial participation with label skew, as measured above. FedAvg's client sampling is unseeded in NVFlare, so the partial-participation cells are not bitwise reproducible.

## What this does not do

- The server never runs a probe forward pass and never sees raw data; it sees only the
  score vectors a site chose to send. A site that lies about its scores is not caught.
- This is a fault detector for honest degradation, not a defence against an adversary.
- Every number above comes from the NVFlare simulator on one machine.
