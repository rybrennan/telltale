"""Run configuration.

Every number that can change an outcome lives here, and every run is fully
determined by a `FedConfig`. That is a hard requirement, not a convenience:
the deliverable of this harness is a detection probability and a false-alarm
rate, and neither is meaningful if a run cannot be reproduced exactly.

Seeds are split by concern so that one source of variation can be held fixed
while another is swept:

  partition_seed  which client gets which samples (the non-IID draw)
  init_seed       global model initialisation
  train_seed      batch shuffling and any training-time stochasticity

A fault-injection sweep, for example, holds `partition_seed` fixed so that
detector behaviour is attributable to the injected fault rather than to a
different data split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class FedConfig:
    """Everything that determines a run.

    Frozen; `fingerprint()` hashes it into the name of the run's result file.
    """

    # ── federation ────────────────────────────────────────────────────────
    n_clients: int = 10
    rounds: int = 30
    client_fraction: float = 1.0
    """Fraction of clients sampled per round.

    Defaults to 1.0 (all clients every round). This differs from the classic
    FedAvg default of C=0.1 on purpose: the integrity layer observes each
    node's contribution per round, so partial participation would confound
    'node not seen this round' with 'node not diverging this round'. Partial
    participation is a later robustness sweep, not the baseline.
    """

    # ── local training ────────────────────────────────────────────────────
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    momentum: float = 0.9

    # ── data ──────────────────────────────────────────────────────────────
    dataset: str = "fashion_mnist"
    dirichlet_alpha: float = 0.5
    """Concentration of the Dirichlet partition. Lower is more heterogeneous.

    This is the single most important knob in the whole study. Legitimate
    label skew across nodes is exactly what a naive integrity detector will
    mistake for corruption, so the detector must be characterised ACROSS a
    range of alpha, not at one value. alpha=0.5 is the common moderate-skew
    setting in the federated learning literature; alpha>=100 is effectively
    IID; alpha<=0.1 is severe skew.
    """
    min_client_samples: int = 20
    """Reject and redraw a partition that starves any client below this."""

    # ── seeds ─────────────────────────────────────────────────────────────
    partition_seed: int = 0
    init_seed: int = 0
    train_seed: int = 0

    # ── execution ─────────────────────────────────────────────────────────
    device: str = "cpu"
    eval_batch_size: int = 512

    # ── bookkeeping ───────────────────────────────────────────────────────
    tag: str = "baseline"
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject values that cannot describe a federation.

        Checked at construction rather than mid-run.

        Raises:
            ValueError: If `n_clients` < 2, `client_fraction` is outside
                (0, 1], `dirichlet_alpha` <= 0, or `rounds` < 1.
        """
        if self.n_clients < 2:
            raise ValueError("n_clients must be >= 2 for a federation")
        if not 0.0 < self.client_fraction <= 1.0:
            raise ValueError("client_fraction must be in (0, 1]")
        if self.dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be > 0")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")

    @property
    def clients_per_round(self) -> int:
        """Clients that report each round.

        `client_fraction` of `n_clients`, rounded, never below 1.
        """
        return max(1, round(self.n_clients * self.client_fraction))

    def to_dict(self) -> dict:
        """Plain-dict form; the input to `fingerprint`.

        Returns:
            Every field of the config as a plain dict.
        """
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable short hash of the full config.

        Used to name result files so that a run cannot silently overwrite a
        different run, and so a result can always be traced back to the exact
        configuration that produced it.

        Returns:
            The first 12 hex digits of the SHA-256 of the config's
            key-sorted JSON form.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
