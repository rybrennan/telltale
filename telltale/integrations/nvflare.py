"""The integrity layer as an NVIDIA FLARE `ModelAggregator`.

Same interface as the Flower strategy, carried by NVFlare's own channels:

    probe battery out   every client builds the same battery from a shared
                        seed and verifies it by fingerprint; nothing else is
                        distributed
    score vector in     each client scores the battery with its post-training
                        model and returns ONLY the class probabilities,
                        float16, in `FLModel.meta` (see `probe_meta`)
    history on server   the aggregator keeps every round's scores and runs
                        the harness detector over all rounds seen so far
    exclusion hook      off by default; with `exclude_flagged=True` a flagged
                        client's update is left out of the average from then on

The detector is the same code the measurement harness uses: `level_series`
for the within-round divergence from the leave-one-out consensus, and `cusum`
over a fixed reference epoch, accumulated over each client's own contacts. A
client that is not sampled in a round is absent from that round, not zero, so
partial participation needs no special handling here.

Aggregation is a sample-count-weighted average of whatever `params` the
clients send (full weights or differences), weighted by
`NUM_STEPS_CURRENT_ROUND` like NVFlare's built-in FedAvg aggregation. The
server never runs a probe forward pass.

Rounds are numbered from 0 in NVFlare and in the harness, so stored arrays
are directly comparable to the grid's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..detect import cusum, level_series
from .wire import decode_scores, encode_scores

try:
    from nvflare.apis.fl_constant import FLMetaKey
    from nvflare.app_common.abstract.fl_model import FLModel
    from nvflare.app_common.aggregators.model_aggregator import (
        ModelAggregator,
    )
    from nvflare.app_common.app_constant import AppConstants
except ImportError as exc:  # pragma: no cover
    # Reached only when nvflare is not installed.
    raise ImportError(
        "telltale.integrations.nvflare requires `pip install nvflare`"
    ) from exc


PROBE_KEY = "telltale_probe_scores"
FINGERPRINT_KEY = "telltale_battery_fingerprint"
CLIENT_NAME_KEY = "client_name"
FLAGGED_KEY = "telltale_flagged"


def probe_meta(probs: np.ndarray, battery_fingerprint: str) -> dict[str, Any]:
    """Build the two `FLModel.meta` entries a client adds to its result.

    Args:
        probs: (n_probes, n_classes) output of `score_probes` on the
            post-training model.
        battery_fingerprint: Fingerprint of the battery that was scored.

    Returns:
        The probe scores as float16 bytes (about 1 kB for 50 probes and 10
        classes) and the battery fingerprint, keyed for the aggregator.
    """
    return {
        PROBE_KEY: encode_scores(probs),
        FINGERPRINT_KEY: battery_fingerprint,
    }


class IntegrityAggregator(ModelAggregator):
    """Weighted FedAvg with the aggregation-integrity layer in front.

    Every constructor argument is a plain JSON type and is kept as an
    attribute of the same name: NVFlare exports a component by reading its
    constructor arguments back from attributes and drops any it cannot find.

    Args:
        n_probes: Number of probes in the battery.
        n_classes: Number of classes per probe.
        battery_fingerprint: Fingerprint every client must echo.
        num_rounds: Rounds in the job; sizes the history array.
        max_clients: Distinct clients the history can hold. A client named
            `site-k` gets row k - 1; any other name gets the next free row.
        reference_start: First round of the CUSUM reference epoch.
        reference_stop: Round after the last round of the reference epoch.
        threshold: CUSUM threshold. A client above it in the current round is
            flagged. None records without flagging.
        exclude_flagged: Leave every client flagged so far out of the
            average. If that would exclude every update, all are kept.
        history_path: Where to write the detector history as JSON after each
            round. None writes nothing.
    """

    def __init__(
        self,
        *,
        n_probes: int,
        n_classes: int,
        battery_fingerprint: str,
        num_rounds: int,
        max_clients: int,
        reference_start: int = 5,
        reference_stop: int = 15,
        threshold: float | None = None,
        exclude_flagged: bool = False,
        history_path: str | None = None,
    ) -> None:
        super().__init__()
        self.n_probes = n_probes
        self.n_classes = n_classes
        self.battery_fingerprint = battery_fingerprint
        self.num_rounds = num_rounds
        self.max_clients = max_clients
        self.reference_start = reference_start
        self.reference_stop = reference_stop
        self.threshold = threshold
        self.exclude_flagged = exclude_flagged
        self.history_path = history_path

        self.client_index: dict[str, int] = {}
        self.probs: dict[int, dict[int, np.ndarray]] = {}
        self.flagged: dict[int, list[str]] = {}
        self.excluded: dict[int, list[str]] = {}
        self.participants: dict[int, list[str]] = {}
        self.fingerprint_mismatches: list[tuple[int, str]] = []
        self.missing_scores: list[tuple[int, str]] = []
        self.payload_bytes: list[int] = []
        self._ever_flagged: set[str] = set()
        self._round = -1
        self._pending: list[tuple[str, FLModel]] = []

    @property
    def reference(self) -> slice:
        """The CUSUM reference epoch, in rounds."""
        return slice(self.reference_start, self.reference_stop)

    # ── per-round state ───────────────────────────────────────────────────
    def reset_stats(self) -> None:
        """Start a new round: drop buffered updates, advance the round."""
        self._pending = []
        self._round = self._current_round()

    def _current_round(self) -> int:
        """Return the controller's round, else one past the last round."""
        if self.fl_ctx is not None:
            r = self.fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            if r is not None:
                return int(r)
        return self._round + 1

    def _new_index(self, name: str) -> int:
        """Return the history row for a newly seen client.

        Pinning `site-k` to row k - 1 keeps a client on the same row in every
        run, whatever order clients first report in, so row k here is shard
        k in the harness.

        Args:
            name: The client's site name.

        Returns:
            Row k - 1 for `site-k` when free, else the lowest free row.

        Raises:
            ValueError: If more than `max_clients` distinct clients report.
        """
        used = set(self.client_index.values())
        if len(used) >= self.max_clients:
            raise ValueError(
                f"more than max_clients={self.max_clients} clients reported"
            )
        tail = name.rsplit("-", 1)[-1]
        if tail.isdigit():
            row = int(tail) - 1
            if 0 <= row < self.max_clients and row not in used:
                return row
        return min(set(range(self.max_clients)) - used)

    def accept_model(self, model: FLModel) -> None:
        """Buffer one client's update and record its probe scores.

        Updates are buffered rather than summed on arrival because whether a
        client is excluded is only known once every client's scores for the
        round are in.

        Args:
            model: The client's result, with the `probe_meta` entries.
        """
        meta = model.meta or {}
        name = str(meta.get(CLIENT_NAME_KEY, f"client-{len(self._pending)}"))
        self._pending.append((name, model))
        r = self._round
        self.participants.setdefault(r, []).append(name)
        if name not in self.client_index:
            self.client_index[name] = self._new_index(name)
        if PROBE_KEY not in meta:
            self.missing_scores.append((r, name))
            return
        if meta.get(FINGERPRINT_KEY) != self.battery_fingerprint:
            # Instrument self-check: a client scoring a different battery is
            # an instrument fault, not a platform fault. Record, don't score.
            self.fingerprint_mismatches.append((r, name))
            return
        blob = meta[PROBE_KEY]
        self.payload_bytes.append(len(blob))
        row = self.client_index[name]
        self.probs.setdefault(r, {})[row] = decode_scores(
            blob, self.n_probes, self.n_classes
        )

    # ── the detector, over everything seen so far ─────────────────────────
    def probs_array(self) -> np.ndarray:
        """Return every round's scores as one dense array.

        Returns:
            (rounds, clients, probes, classes) float32, NaN where a client
            was absent or sent no usable scores.
        """
        shape = (
            self.num_rounds,
            self.max_clients,
            self.n_probes,
            self.n_classes,
        )
        out = np.full(shape, np.nan, dtype=np.float32)
        for r, by_client in self.probs.items():
            if 0 <= r < self.num_rounds:
                for k, p in by_client.items():
                    out[r, k] = p
        return out

    def cusum_so_far(self) -> np.ndarray:
        """Return the CUSUM of the level statistic over the rounds seen.

        Returns:
            (rounds, clients) CUSUM, NaN in the future, through the reference
            epoch, and everywhere if the run cannot hold a two-round epoch.
        """
        if len(range(self.num_rounds)[self.reference]) < 2:
            return np.full((self.num_rounds, self.max_clients), np.nan)
        level = level_series(self.probs_array())["level"]
        return cusum(level, self.reference)

    def _flag(self, r: int) -> list[str]:
        """Return the clients whose CUSUM exceeds the threshold in round r."""
        if self.threshold is None or r >= self.num_rounds:
            return []
        c = self.cusum_so_far()
        names = {k: n for n, k in self.client_index.items()}
        return sorted(
            names[k]
            for k in names
            if np.isfinite(c[r, k]) and c[r, k] > self.threshold
        )

    def aggregate_model(self) -> FLModel:
        """Flag, optionally exclude, then average the kept updates.

        Returns:
            The weighted average, with this round's flagged clients under
            `FLAGGED_KEY` in its meta.
        """
        r = self._round
        flagged = self._flag(r)
        self.flagged[r] = flagged
        self._ever_flagged.update(flagged)
        if flagged:
            self.warning(f"telltale: round {r} flagged {flagged}")

        kept = self._pending
        if self.exclude_flagged and self._ever_flagged:
            kept = [
                (n, m) for n, m in self._pending if n not in self._ever_flagged
            ]
            reporting = {n for n, _ in self._pending}
            self.excluded[r] = sorted(reporting & self._ever_flagged)
            if not kept:  # never aggregate nothing; fall back to the full set
                kept = self._pending
        aggregated = self._weighted_average(kept)
        aggregated.meta = {FLAGGED_KEY: flagged}
        self._write_history()
        return aggregated

    @staticmethod
    def _weighted_average(updates: list[tuple[str, FLModel]]) -> FLModel:
        """Average `params` weighted by NUM_STEPS_CURRENT_ROUND.

        Args:
            updates: (client name, result) pairs; a result without
                NUM_STEPS_CURRENT_ROUND weighs 1.0.

        Returns:
            The weighted average, with the clients' params_type.

        Raises:
            ValueError: If params types differ or the weights sum to zero.
        """
        if not updates:
            return FLModel(params={})
        params_type = updates[0][1].params_type
        total = 0.0
        acc: dict[str, Any] = {}
        for _, m in updates:
            if m.params_type != params_type:
                raise ValueError(
                    f"params_type mismatch: {params_type} vs {m.params_type}"
                )
            meta = m.meta or {}
            w = float(meta.get(FLMetaKey.NUM_STEPS_CURRENT_ROUND, 1.0))
            total += w
            for key, value in m.params.items():
                acc[key] = acc[key] + value * w if key in acc else value * w
        if total <= 0:
            raise ValueError("aggregation weights sum to zero")
        params = {k: v / total for k, v in acc.items()}
        return FLModel(params=params, params_type=params_type)

    # ── what was observed, for scoring after the run ──────────────────────
    def history(self) -> dict[str, Any]:
        """Return everything the aggregator observed, as JSON-ready types."""
        c = self.cusum_so_far()
        return {
            "client_index": self.client_index,
            "reference": [self.reference_start, self.reference_stop],
            "threshold": self.threshold,
            "exclude_flagged": self.exclude_flagged,
            "participants": {str(r): v for r, v in self.participants.items()},
            "flagged": {str(r): v for r, v in self.flagged.items()},
            "excluded": {str(r): v for r, v in self.excluded.items()},
            "fingerprint_mismatches": self.fingerprint_mismatches,
            "missing_scores": self.missing_scores,
            "payload_bytes_median": (
                int(np.median(self.payload_bytes))
                if self.payload_bytes
                else None
            ),
            "cusum": [
                [float(v) if np.isfinite(v) else None for v in row] for row in c
            ],
            "probs": {
                str(r): {str(k): p.tolist() for k, p in by.items()}
                for r, by in self.probs.items()
            },
        }

    def _write_history(self) -> None:
        """Write `history()` to `history_path`, if one was given."""
        if self.history_path:
            path = Path(self.history_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.history()))
