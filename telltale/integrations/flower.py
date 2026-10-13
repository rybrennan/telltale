"""The integrity layer as a Flower strategy.

Interface, as promised in the solution brief:

    probe battery out   every node builds the same battery from a shared seed
                        and verifies it by fingerprint; nothing else is
                        distributed
    score vector in     each node scores the battery with its post-training
                        model and returns ONLY the class probabilities,
                        float16, in Flower's FitRes metrics channel
    quarantine hook     a flagged node's update can be excluded from
                        aggregation

The detector is the same code the measurement harness uses: `level_series` for
the within-round divergence from the leave-one-out consensus, and `cusum` over a
fixed reference epoch of rounds. Nothing here reimplements a statistic.

What this module does NOT do: it does not score raw sensor data, and the server
never runs a probe forward pass. The server only ever sees the score vectors a
node chose to send, which is the property the multi-domain argument rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..detect import cusum, level_series
from .wire import decode_scores, encode_scores  # noqa: F401  (re-exported)

try:
    from flwr.common import FitRes, Parameters, Scalar
    from flwr.server.client_proxy import ClientProxy
    from flwr.server.strategy import FedAvg
except ImportError as exc:  # pragma: no cover
    # Reached only when flwr is not installed.
    raise ImportError(
        "telltale.integrations.flower requires `pip install flwr`"
    ) from exc


PROBE_KEY = "probe_scores"
FINGERPRINT_KEY = "battery_fingerprint"
NODE_KEY = "node_id"


@dataclass
class IntegrityState:
    """Everything the strategy observed, for scoring after the run."""

    probs: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    """probs[round][node] -> (n_probes, n_classes)"""
    payload_bytes: list[int] = field(default_factory=list)
    flagged: dict[int, list[int]] = field(default_factory=dict)
    """round -> nodes whose CUSUM exceeded the threshold that round"""
    quarantined: dict[int, list[int]] = field(default_factory=dict)
    fingerprint_mismatches: list[tuple[int, int]] = field(default_factory=list)

    def probs_array(
        self, n_rounds: int, n_nodes: int, n_probes: int, n_classes: int
    ) -> np.ndarray:
        """Stack the recorded scores into the layout `detect` consumes.

        Args:
            n_rounds: Number of rounds.
            n_nodes: Number of nodes.
            n_probes: Number of probes in the battery.
            n_classes: Number of classes.

        Returns:
            Dense (rounds, nodes, probes, classes) float32 array, NaN where a
            node did not report or reported a mismatched battery.
        """
        out = np.full(
            (n_rounds, n_nodes, n_probes, n_classes), np.nan, dtype=np.float32
        )
        for r, by_node in self.probs.items():
            for k, p in by_node.items():
                out[r, k] = p
        return out


class IntegrityFedAvg(FedAvg):
    """FedAvg with the aggregation-integrity layer in front of the average.

    Flower numbers rounds from 1; the harness numbers them from 0. Everything
    stored here uses the harness convention (flower_round - 1) so the stored
    array is directly comparable to the grid's.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        n_probes: int,
        n_classes: int,
        battery_fingerprint: str,
        n_rounds: int,
        reference: slice = slice(5, 15),
        threshold: float | None = None,
        quarantine: bool = False,
        **fedavg_kwargs: Any,
    ) -> None:
        """Set up the strategy and an empty `IntegrityState`.

        Args:
            n_nodes: Number of nodes in the federation.
            n_probes: Number of probes in the battery.
            n_classes: Number of classes.
            battery_fingerprint: Fingerprint every node must echo; a report
                with any other value is an instrument mismatch.
            n_rounds: Total rounds in the run.
            reference: The CUSUM reference epoch, in harness (0-based)
                rounds.
            threshold: With this set, a node whose CUSUM exceeds it is
                flagged. None disables flagging.
            quarantine: With `threshold` as well, a flagged node's update is
                excluded from aggregation from that round on.
            **fedavg_kwargs: Passed through to Flower's FedAvg.
        """
        super().__init__(**fedavg_kwargs)
        self.n_nodes = n_nodes
        self.n_probes = n_probes
        self.n_classes = n_classes
        self.battery_fingerprint = battery_fingerprint
        self.n_rounds = n_rounds
        self.reference = reference
        self.threshold = threshold
        self.quarantine = quarantine
        self.state = IntegrityState()
        self._ever_flagged: set[int] = set()

    # ── the detector, over everything seen so far ────────────────────────────
    def cusum_so_far(self) -> np.ndarray:
        """Compute the CUSUM of the level statistic over every round so far.

        Returns:
            (rounds, nodes) CUSUM, NaN in rounds not yet run.
        """
        arr = self.state.probs_array(
            self.n_rounds, self.n_nodes, self.n_probes, self.n_classes
        )
        return cusum(level_series(arr)["level"], self.reference)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Record probe scores, flag, optionally quarantine, then run FedAvg.

        A node whose battery fingerprint differs is logged as an instrument
        mismatch and contributes no scores. If quarantine would exclude every
        node the full set is aggregated instead.

        Args:
            server_round: Flower round number, starting at 1.
            results: (client, fit result) pairs; each result's metrics carry
                the node id, battery fingerprint and probe scores.
            failures: Failed fits, passed through to FedAvg.

        Returns:
            FedAvg's (parameters, metrics), with `integrity_flagged`
            (comma-joined node ids) added to the metrics.
        """
        r = server_round - 1
        self.state.probs.setdefault(r, {})
        node_of: dict[int, int] = {}
        for i, (_, res) in enumerate(results):
            m = res.metrics
            node = int(m[NODE_KEY])
            node_of[i] = node
            if m.get(FINGERPRINT_KEY) != self.battery_fingerprint:
                # Instrument self-check: a node scoring a different battery
                # is an instrument fault, not a platform fault. Record, don't
                # score.
                self.state.fingerprint_mismatches.append((r, node))
                continue
            blob = m[PROBE_KEY]
            self.state.payload_bytes.append(len(blob))
            self.state.probs[r][node] = decode_scores(
                blob, self.n_probes, self.n_classes
            )

        flagged: list[int] = []
        if self.threshold is not None:
            c = self.cusum_so_far()
            flagged = [
                k
                for k in range(self.n_nodes)
                if np.isfinite(c[r, k]) and c[r, k] > self.threshold
            ]
            self._ever_flagged.update(flagged)
        self.state.flagged[r] = flagged

        kept = results
        if self.quarantine and self._ever_flagged:
            kept = [
                res
                for i, res in enumerate(results)
                if node_of[i] not in self._ever_flagged
            ]
            self.state.quarantined[r] = sorted(self._ever_flagged)
            if not kept:  # never aggregate nothing; fall back to the full set
                kept = results
        params, metrics = super().aggregate_fit(server_round, kept, failures)
        metrics = dict(metrics or {})
        metrics["integrity_flagged"] = ",".join(map(str, flagged))
        return params, metrics
