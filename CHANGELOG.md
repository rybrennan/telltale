# Changelog

Stages were tagged in the private working repository. This public repository is a single squashed release (`v0.6.0`); each entry below states what that stage claimed and what it did not.

## v0.7.0 (2026-10-13): the layer as an NVIDIA FLARE aggregator
- `telltale.integrations.nvflare.IntegrityAggregator`, an NVFlare `ModelAggregator`: sites return
  probe scores (float16, 1,000 bytes) in `FLModel.meta`; the aggregator keeps the detector history
  on the server and runs the same `level_series` and `cusum`; client exclusion is off by default.
  Built on the released NVFlare 2.7.3 and checked on 2.9.0. Design from NVIDIA/NVFlare#5398.
- `examples/nvflare/`: a `FedAvgRecipe` job and a Client API training script, laid out like
  NVFlare's own examples.
- `scripts/nvflare_eval.py`: 72 simulator runs (six seeds per cell). Every site reporting: blur
  caught in 6 of 6 seeds at every heterogeneity level, 0 to 9 rounds after onset, at most 0.7%
  held-out false alarms. Six of ten sites sampled per round: 6 of 6 near-IID with false alarms
  at 2.7%, 4 of 6 at moderate skew, 1 of 6 at severe skew, because the rotating peer set makes
  a healthy site's divergence more variable under skew. Pooling the reference spread did not
  recover it; a consensus frozen at the reference epoch did, at the cost of false alarms, and
  is not adopted.
- The probe-score wire format moves to `telltale/integrations/wire.py`, shared by both adapters.
  `pyproject.toml` makes the package installable (`pip install -e ".[nvflare]"`).
- Eleven integration tests, in the repository's Google style and ruff-clean. 113 tests in
  total; CI installs NVFlare.

## Unreleased: Google Python style pass (no behaviour change)
- Every line at 80 columns or fewer; Google-style docstrings with Args, Returns and
  Raises on the library's public functions; type hints on every signature.
- `pyproject.toml` holds the ruff configuration, and CI now runs `ruff check`.
- Verified behaviour-preserving: all 102 tests pass, and a golden master over 295
  outputs (seeded training runs, partitions, CUSUM, and every result script, figure
  and results file over the 276 probe arrays) is byte-identical before and after.

## v0.6.0 (2026-09-23): the layer as a Flower strategy
- `telltale.integrations.flower.IntegrityFedAvg`, a Flower `FedAvg` subclass: nodes return
  probe scores (float16, about 1 kB) in the fit-result metrics; the strategy runs the same
  `level_series` and `cusum` functions as the harness and can quarantine flagged nodes.
- `scripts/flower_demo.py`: five client processes over gRPC. One slow drift, flagged on the right
  node seven rounds after onset, no healthy node flagged, 1,000 bytes per node per round. One run
  per condition: a demonstration, not a detection rate.
- Six integration tests, including one that pins the known limit (drift toward the fleet is not
  flagged by the one-sided statistic). 96 tests in total.

## v0.5.0 (2026-09-09): acoustic grid scored; node confound resolved; submission hygiene
- DeepShip grid (99 runs): the ordering by heterogeneity transfers; CUSUM transfers,
  persistence does not; gain is caught here; the fleet metric is uninformative rather than
  hidden; severe skew catches nothing.
- Node dimension (72 runs): node-independent at near-IID and moderate skew; node-dependent
  at severe skew.
- Licence, citation, lock file, CI, Dockerfile, reproduce script, tracked result summaries.

## v0.4.0 (2026-09-09): second modality, submarine schedule, threshold sweep
- DeepShip underwater acoustics through the unchanged harness (`telltale/acoustic.py`).
- Contact schedules: submarine nodes report every k rounds; temporal statistics run over a
  node's own contacts; time to detection in contacts.
- Thresholds per platform class; CUSUM reference spread pooled where a class is starved.
- Two-sided CUSUM judged on healthy runs; threshold sweep per statistic per alpha;
  `results/scores.json` carries every number in the README's tables.
- Overnight grids: node dimension (clothing) and the full fault grid (acoustics).

## v0.3.0 (2026-09-09): stage 4: the grid
- 99-run grid (3 alphas × 3 seeds × healthy + 10 faults), 10 h, scored into detection
  probability with Wilson intervals and a chance floor; false-alarm rate on held-out seeds.
- Fleet metric hid all 90 faults; label corruption localised at every alpha; slow drift under
  skew caught by CUSUM alone; gain the failure case.

## v0.2.0 (2026-09-08): stages 2 and 3: instrument and faults
- Probe battery, monitor on the `update_hook` seam proven bitwise inert at scale; healthy
  divergence floor per alpha and battery size; bandwidth knee at ~1 kB.
- Fault injection with known ground truth through a second seam; metrics for Pd / FAR / ttd.

## v0.1.0 (2026-09-07): stage 1: clean baseline
- FedAvg implemented directly, Dirichlet non-IID partitioning, seeds split by concern,
  bitwise reproducibility asserted. Three alpha arms; the noise floor scales with
  heterogeneity 3.4×. No detector, no faults, no detection claim.
