# telltale with FEDn server functions

Cross-round client monitoring inside a FEDn session. Each client trains on its
own data, scores a fixed 50-image probe set with its post-training model, and
puts the class probabilities in its training metadata (float16, 1,000 bytes,
sent as hex). The monitor runs inside FEDn's `ServerFunctions.aggregate`. It
judges each client only on those probe answers, never on its model update. It
measures how far a client's answers are from the other clients' and
accumulates that over the client's own early rounds with a CUSUM. It flags
only. Aggregation is ordinary weighted FedAvg over every client.

| Piece | Where |
| --- | --- |
| The monitor | `server_functions.py`, one class, numpy only |
| Probe reports from clients | `client_train.py`, in the training metadata |
| Same numbers as telltale's detector | `tests/test_fedn_adapter.py` (zero difference on a synthetic fleet) |
| Test rig and results | `run_experiment.py`, `run_batch.sh`, `score.py`, `results/` |

FEDn runs server functions from the class source alone and offers a short
import allowlist. The class therefore carries its own small JSON writer and
uses nothing but numpy and FEDn's logger. Each round it writes one
`TELLTALE {...}` line to the hook service log with every client's level and
CUSUM. Set `THRESHOLD` to get a `TELLTALE_FLAG` line as well.

## What was tested

FEDn v0.33.0 (Apache-2.0, the last open release, September 2025), run with
its own Docker Compose stack on one machine. This is not Scaleout Edge 1.x,
and nothing here claims it works there.

Six clients, MNIST split six ways, 40 rounds. Five healthy runs and four runs
with one faulty client. Faults start at round 20 and ramp over 10 rounds.
One run per fault.

The threshold is the highest CUSUM in three healthy runs. Every way of
choosing three of the five was scored (10 rotations); the other two healthy
runs measure false alarms.

| Fault run | Right client named | Innocent client or early flag | When |
| --- | --- | --- | --- |
| label noise 30%, client 3 | 10 of 10 | 10 of 10 (rule 1), 1 of 10 (rule 2) | onset round |
| label noise 30%, client 6 | 10 of 10 | 10 of 10 (rule 1), 0 of 10 (rule 2) | 1 round after onset |
| blur sigma 1.0, client 2 | 10 of 10 (rule 1), 9 of 10 (rule 2) | 1 of 10 | 10 to 12 rounds after onset |
| blur sigma 1.0, client 5 | 10 of 10 | 0 of 10 | 17 to 19 rounds after onset |

Rule 1 flags any client above the threshold. Rule 2 also requires the client
to be an outlier among that round's CUSUMs (robust z above 3.5, the standard
Iglewicz-Hoaglin cutoff, not tuned). Rule 2 lives in `score.py`, not in the
monitor.

What to take from it, and what not to:

- Under label noise the faulty client ends 2.9 times (client 6) and 11.5
  times (client 3) higher than any other. But every client is compared with the mean of the others, so one
  bad client pulls that mean and the healthy clients rise too. With six
  clients rule 1 flags them every time. This is the reason rule 2 exists.
- Rule 2 was written after seeing the client-3 label-noise run and before the
  other three fault runs had finished. Only those three are evidence for it.
- Client 5's blur was caught barely: a peak of 7.14 against a healthy maximum
  of 6.94.
- 5 of 20 held-out healthy runs had at least one alarm. A threshold set from
  three healthy runs gives one in four by construction. More calibration runs
  lower it.

## Run it

1. Start FEDn v0.33.0 from its repository
   (`git clone https://github.com/scaleoutsystems/scaleout-client` and check
   out tag `v0.33.0`). The MinIO image named in its compose file can no longer
   be pulled; `compose.local.yaml` here swaps it:

   ```bash
   docker compose -p fedn033 -f docker-compose.yaml -f <this dir>/compose.local.yaml up -d minio mongo api-server hooks combiner
   ```

   Build the client image once with the upstream server-functions example
   (`examples/server-functions/docker-compose.override.yaml`, service
   `client1`); `compose.clients.yaml` reuses that image as `fedn033-client1`.

2. Make the client package: copy FEDn's `examples/server-functions/client`
   folder, replace its `train.py` with `client_train.py`, then
   `fedn package create --path client` and `fedn run build --path client` for
   the seed model. Note: `fedn run build` fails if the path contains a space.

3. Upload the package and seed with `APIClient.set_active_package(...,
   helper="numpyhelper", name=...)` and `set_active_model("seed.npz")`, then
   start six clients: `docker compose -f compose.clients.yaml up -d`.

4. Run the batch and score it:

   ```bash
   PY=python ./run_batch.sh
   python score.py
   ```

   Edit `SEED_MODEL` in `run_batch.sh` to your seed model's id. Each run took
   about seven minutes on one laptop.

Two platform details matter for anyone deploying this:

- FEDn reuses the server-functions instance while the code string is
  unchanged, so the monitor's history carries from one session to the next.
  The rig sets a unique `RUN_TAG` per run to start clean.
- FEDn v0.33.0 runs the submitted class without enforcing the import
  allowlist. The monitor stays inside it anyway.
