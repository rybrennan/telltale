"""Contact schedules: which nodes report in which rounds.

A surface ship or shore station reports every round. A submarine reports
only when it has a communications window. The schedule is the third seam on
`run_federated`: `participation(round_idx) -> list[int]`. The federation code
knows nothing about platforms.

Model of a silent round, stated so it is not mistaken for more than it is: a
node that is out of contact does nothing that round: it does not train, does
not report, and on its next contact it receives the current global model. That
is the "surface, sync, train, report" case. It is NOT the case where a boat
keeps training on a stale model while submerged and returns one large update
weeks later; that is asynchronous aggregation, a research question in its own
right, and out of scope for this harness (docs/stage3_fault_design.md s.7).

What this does let the harness measure: every temporal statistic in `detect`
is computed over a node's *own observed rounds*, so a boat is compared with its
last contact rather than with a round it was silent for, and time to detection
is counted in contacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ContactSchedule:
    """Which nodes report in which rounds.

    `sub_nodes` report every `period` rounds, every other node every round.
    Frozen and fingerprintable, so a result names its schedule.
    """

    period: int = 1
    """A submarine node reports every `period` rounds. 1 means every round."""
    sub_nodes: tuple[int, ...] = ()
    """Nodes on the submarine schedule.

    Every other node reports every round. Contacts are staggered: the i-th
    submarine's phase is i mod period.
    """

    def __post_init__(self) -> None:
        """Validate the schedule.

        Raises:
            ValueError: If the period is below 1, or sub-node ids repeat or
                are negative.
        """
        if self.period < 1:
            raise ValueError("period must be >= 1")
        if len(set(self.sub_nodes)) != len(self.sub_nodes) or any(
            n < 0 for n in self.sub_nodes
        ):
            raise ValueError("sub_nodes must be distinct and >= 0")

    def phase(self, node: int) -> int:
        """Return the offset that staggers submarine `node`'s contacts.

        The offset is its position in `sub_nodes` mod `period`, so boats do
        not all surface in the same round.

        Args:
            node: A node id in `sub_nodes`.

        Returns:
            The phase in [0, period).

        Raises:
            ValueError: If `node` is not in `sub_nodes`.
        """
        return self.sub_nodes.index(node) % self.period

    def in_contact(self, node: int, round_idx: int) -> bool:
        """Return whether `node` reports in `round_idx`.

        Always true for a non-submarine; every `period`-th round (shifted by
        its phase) for a submarine.

        Args:
            node: Node id.
            round_idx: 0-based round index.

        Returns:
            True if the node is in contact that round.
        """
        if node not in self.sub_nodes:
            return True
        return (round_idx + self.phase(node)) % self.period == 0

    def participants(self, round_idx: int, n_clients: int) -> list[int]:
        """Return the ids that report in `round_idx`.

        This is the `participation` seam of `run_federated`.

        Args:
            round_idx: 0-based round index.
            n_clients: Number of clients; ids are range(n_clients).

        Returns:
            The participating client ids in ascending order.
        """
        return [c for c in range(n_clients) if self.in_contact(c, round_idx)]

    def observed(self, n_rounds: int, n_clients: int) -> np.ndarray:
        """Return the contact mask for a whole run.

        Args:
            n_rounds: Number of rounds.
            n_clients: Number of clients.

        Returns:
            (rounds, clients) bool mask, True where a node reports.
        """
        return np.array(
            [
                [self.in_contact(c, r) for c in range(n_clients)]
                for r in range(n_rounds)
            ],
            dtype=bool,
        )

    def to_dict(self) -> dict:
        """Return the plain-dict form.

        Returns:
            The dataclass fields as a dict with `sub_nodes` as a list; the
            input to `fingerprint`.
        """
        d = asdict(self)
        d["sub_nodes"] = list(self.sub_nodes)
        return d

    def fingerprint(self) -> str:
        """Return a stable hash of the schedule.

        Returns:
            The first 12 hex digits of the SHA-256 of the sorted-key JSON.
        """
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()
        ).hexdigest()[:12]
