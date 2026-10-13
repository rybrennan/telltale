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

`scripts/nvflare_eval.py`, 36 simulator runs (about 50 minutes, three at a time on one
laptop CPU). 10 sites, 30 rounds, Fashion-MNIST, a 50-probe battery. Fault: site-3's images
blur, ramping to 1.5 px over ten rounds from round 15. Thresholds are the CUSUM healthy
maximum after the reference epoch (rounds 5-14) on healthy seed 0, per cell, and never see a
fault. False alarms are counted on held-out healthy seeds 1 and 2, per client contact. Three
fault seeds per cell, so detection is "caught in every seed tried," with a Wilson 95%
interval, not a probability. Summary: `results/nvflare_eval.json`.

| Label skew (alpha) | Sites sampled per round | Threshold | False alarms, held-out contacts | Blur caught | Rounds from onset to flag (client contacts) |
| --- | --- | --- | --- | --- | --- |
| 100 (near-IID) | 10 of 10 | 8.76 | 0 / 300 | 3 / 3 [0.44, 1.00] | 6, 6, 7 (7, 7, 8) |
| 100 (near-IID) | 6 of 10 | 7.63 | 0 / 180 | 3 / 3 [0.44, 1.00] | 6, 10, 10 (6, 5, 7) |
| 0.5 (moderate) | 10 of 10 | 3.77 | 0 / 300 | 3 / 3 [0.44, 1.00] | 5, 9, 4 (6, 10, 5) |
| 0.5 (moderate) | 6 of 10 | 6.62 | 0 / 180 | 1 / 3 [0.06, 0.79] | 8 (3) |
| 0.1 (severe) | 10 of 10 | 0.71 | 3 / 300 (1.0%) | 3 / 3 [0.44, 1.00] | 6, 1, 4 (7, 2, 5) |
| 0.1 (severe) | 6 of 10 | 6.54 | 0 / 180 | 0 / 3 [0.00, 0.56] | none |

With every site reporting, the slow blur is caught in every seed at every heterogeneity
level, one to nine rounds after onset (the one-round case is discussed below), with at most
one healthy site also flagged in a run.
This matches the in-process harness (CUSUM, blur 1.5 px: caught in 3 of 3 seeds at every
alpha). The one-round detection at alpha 0.1 is in the cell whose threshold is lowest; at a
1% per-contact false-alarm rate, the chance of a false flag on that one site somewhere in the
fifteen post-onset rounds is about 14%, so it is reported, not leaned on. Its CUSUM kept
rising, from 1.9 to 19.2 over the next eight rounds.

**Partial participation costs detection under label skew, and the cost is measured, not
assumed.** Near-IID it costs nothing. At moderate and severe skew the same fault is caught in
1 and 0 of 3 seeds. Two things change when FedAvg samples 6 of 10 sites each round: each site
has as few as 3 contacts in the reference epoch instead of 10, and a site's leave-one-out
consensus is built from a different set of peers every round. The evidence points at the
second. The healthy reference-epoch coefficient of variation of the level goes from 0.185
to 0.231 at alpha 0.5 and from 0.081 to 0.153 at alpha 0.1, the per-cell thresholds rise
with it (3.77 to 6.62, 0.71 to 6.54), and a 1.5 px blur no longer clears them. Pooling the
reference spread across sites, the harness's remedy for a starved reference, was tried on the
recorded histories afterwards and did not recover it (moderate skew: 1 of 3), so the sparse
reference alone is not the limit. Building the consensus from a fixed peer set, or scoring a
site against its own history only, are the next things to measure.

Sampling in these runs is NVFlare's own: FedAvg draws a fresh random subset each round and
does not take a seed, so the 60% cells are not bitwise reproducible. The sites sampled each
round are recorded in every history file.

## What this does not do

- The server never runs a probe forward pass and never sees raw data; it sees only the
  score vectors a site chose to send. A site that lies about its scores is not caught.
- This is a fault detector for honest degradation, not a defence against an adversary.
- Every number above comes from the NVFlare simulator on one machine.
