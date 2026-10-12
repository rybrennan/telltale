"""Probe batteries: the instrument that ships with the global model.

A probe battery is a small fixed set of inputs distributed alongside the global
model. Each node scores it with its locally trained model and returns only the
resulting class-probability rows. The battery is identical across nodes and
fixed across rounds, which is what makes both cross-node and across-time
comparison valid: every node answers the same questions every round.

Design decisions and rejected alternatives are recorded in
docs/stage2_probe_design.md. In brief: probes are drawn from the held-out
test split so no node has trained on them; they are stratified by class and
ordered round-robin so any prefix of length k * n_classes is itself an exactly
stratified battery; labels stay on the server and are never sent.

The return payload is `n_probes * n_classes` scalars, independent of model
size and of local data volume.

`ProbeMonitor` attaches to the `update_hook` seam in `run_federated`. It must
not alter the federation it observes: its template model is built under
`torch.random.fork_rng`, scoring runs under `no_grad`, and a test asserts the
final parameters are bitwise identical with and without it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from .data import Dataset
from .fedavg import ClientUpdate, StateDict
from .models import build_model

BYTES_PER_SCALAR = {"float32": 4, "float16": 2, "uint8": 1}


# ── configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeConfig:
    """Everything that determines a battery.

    Separate from FedConfig on purpose: the detector is a layer on top of the
    federation, and adding fields to FedConfig would change the fingerprint of
    every existing baseline run.
    """

    n_probes: int = 500
    probe_seed: int = 0
    source: str = "test"
    """Split the battery is drawn from.

    'test' by default: no node trains on it.
    """
    scalar_dtype: str = "float16"
    """Encoding assumed for the return payload when accounting bytes."""

    def __post_init__(self) -> None:
        """Validate the config.

        Raises:
            ValueError: If the battery is empty, the source split is unknown,
                or the wire dtype is unknown.
        """
        if self.n_probes < 1:
            raise ValueError("n_probes must be >= 1")
        if self.source not in ("test", "train"):
            raise ValueError("source must be 'test' or 'train'")
        if self.scalar_dtype not in BYTES_PER_SCALAR:
            raise ValueError(
                f"scalar_dtype must be one of {list(BYTES_PER_SCALAR)}"
            )

    def to_dict(self) -> dict:
        """Return the plain-dict form.

        Returns:
            The dataclass fields as a dict; the input to `fingerprint`.
        """
        return asdict(self)

    def fingerprint(self) -> str:
        """Return a stable hash of the config.

        Returns:
            The first 12 hex digits of the SHA-256 of the sorted-key JSON.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── the battery ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeBattery:
    """The fixed probe inputs served to every node.

    Only `x` leaves the server.
    """

    x: torch.Tensor
    """(n, 1, 28, 28): the only thing distributed to nodes."""
    y: torch.Tensor
    """(n,) labels. Server-side metadata for diagnostics; never sent."""
    indices: np.ndarray
    """Positions in the source split, so the battery can be traced back."""
    n_classes: int
    source: str
    seed: int
    fingerprint: str = field(default="")

    @property
    def n_probes(self) -> int:
        """Return the number of probe inputs."""
        return int(self.x.shape[0])

    def prefix(self, n: int) -> ProbeBattery:
        """Return the nested sub-battery of the first `n` probes.

        Because probes are ordered round-robin by class, a prefix of length
        k * n_classes is exactly stratified. This is how one run over a large
        battery yields the statistic at every smaller battery size.

        Args:
            n: Number of leading probes to keep, in [1, n_probes].

        Returns:
            A new battery with its own fingerprint over the kept probes.

        Raises:
            ValueError: If `n` is outside [1, n_probes].
        """
        if not 1 <= n <= self.n_probes:
            raise ValueError(f"prefix length {n} not in [1, {self.n_probes}]")
        x, y, idx = self.x[:n], self.y[:n], self.indices[:n]
        return ProbeBattery(
            x=x,
            y=y,
            indices=idx,
            n_classes=self.n_classes,
            source=self.source,
            seed=self.seed,
            fingerprint=battery_fingerprint(x, idx, self.source),
        )


def battery_fingerprint(
    x: torch.Tensor, indices: np.ndarray, source: str
) -> str:
    """Hash the source name, the indices, and the pixel bytes with SHA-256.

    Echoed in every report so the server can confirm a node scored the battery
    it was served, not a stale or altered one.

    Args:
        x: Probe inputs.
        indices: Positions of the probes in the source split.
        source: Name of the source split.

    Returns:
        The first 16 hex digits of the digest.
    """
    h = hashlib.sha256()
    h.update(source.encode())
    h.update(np.ascontiguousarray(indices, dtype=np.int64).tobytes())
    h.update(x.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()[:16]


def build_probe_battery(dataset: Dataset, cfg: ProbeConfig) -> ProbeBattery:
    """Build a stratified, seeded, round-robin-by-class battery.

    Args:
        dataset: Dataset holding the train and test splits.
        cfg: Battery size, seed, and source split.

    Returns:
        The battery drawn from the split named by `cfg.source`.

    Raises:
        ValueError: If some class has too few samples in the split to fill
            its share of the battery.
    """
    if cfg.source == "test":
        x_src, y_src = dataset.test_x, dataset.test_y
    else:
        x_src, y_src = dataset.train_x, dataset.train_y

    n_classes = dataset.n_classes
    y_np = y_src.numpy()
    rng = np.random.default_rng([cfg.probe_seed, 0x9B0BE])
    # One permutation per class, drawn in class order regardless of n_probes,
    # so batteries of different sizes from the same seed are nested prefixes.
    per_class = [
        rng.permutation(np.where(y_np == k)[0]) for k in range(n_classes)
    ]

    indices = np.empty(cfg.n_probes, dtype=np.int64)
    for i in range(cfg.n_probes):
        k, j = i % n_classes, i // n_classes
        if j >= len(per_class[k]):
            raise ValueError(
                f"class {k} has only {len(per_class[k])} samples in the "
                f"{cfg.source} split; cannot build a {cfg.n_probes}-probe "
                "stratified battery"
            )
        indices[i] = per_class[k][j]

    x = x_src[indices].contiguous()
    y = y_src[indices].contiguous()
    return ProbeBattery(
        x=x,
        y=y,
        indices=indices,
        n_classes=n_classes,
        source=cfg.source,
        seed=cfg.probe_seed,
        fingerprint=battery_fingerprint(x, indices, cfg.source),
    )


# ── scoring ───────────────────────────────────────────────────────────────────


@torch.no_grad()
def score_probes(model: nn.Module, battery: ProbeBattery) -> torch.Tensor:
    """Score the battery with `model`.

    Eval mode and no grad, so no training-time randomness is consumed. The
    whole battery is scored in ONE forward pass, deliberately: CPU convolution
    picks different GEMM blocking for different batch sizes, and the same
    model scored in batches of 5 versus 512 differs at the ~3e-8 level. That
    is negligible for any statistic here but breaks bitwise identity, which
    the replay check and the reproducibility guarantee rely on. Fixed
    batching makes every report in a run comparable bit for bit.

    Args:
        model: Model to score with; switched to eval mode.
        battery: Battery whose inputs are scored.

    Returns:
        (n_probes, n_classes) float32 class probabilities.
    """
    model.eval()
    return torch.softmax(model(battery.x), dim=1).to(torch.float32)


def quantise(probs: torch.Tensor | np.ndarray, scalar_dtype: str) -> np.ndarray:
    """Round probabilities to what the wire would carry, returned as float32.

    Applied at analysis time, not in the monitor, so the same reports can be
    compared across encodings.

    Args:
        probs: Class probabilities.
        scalar_dtype: Wire encoding: "float32", "float16" or "uint8".

    Returns:
        The probabilities after a round trip through the encoding.

    Raises:
        ValueError: If `scalar_dtype` is unknown.
    """
    p = np.asarray(probs, dtype=np.float32)
    if scalar_dtype == "float32":
        return p
    if scalar_dtype == "float16":
        return p.astype(np.float16).astype(np.float32)
    if scalar_dtype == "uint8":
        return (np.round(p * 255.0) / 255.0).astype(np.float32)
    raise ValueError(f"unknown scalar_dtype {scalar_dtype!r}")


def payload_bytes(
    n_probes: int, n_classes: int, scalar_dtype: str = "float16"
) -> int:
    """Return the bytes a node sends per round on the probe pathway.

    Args:
        n_probes: Number of probes in the battery.
        n_classes: Number of classes per probability row.
        scalar_dtype: Wire encoding of each scalar.

    Returns:
        n_probes * n_classes * bytes per scalar.
    """
    return n_probes * n_classes * BYTES_PER_SCALAR[scalar_dtype]


def model_bytes(state: StateDict) -> int:
    """Return the bytes a node sends per round for the model update itself.

    Args:
        state: Model state dict.

    Returns:
        Total bytes across all tensors in `state`.
    """
    return sum(v.numel() * v.element_size() for v in state.values())


# ── reports and the monitor ───────────────────────────────────────────────────


@dataclass
class ProbeReport:
    """What one node returns on the probe pathway in one round."""

    client_id: int
    round_idx: int
    battery_fingerprint: str
    probs: torch.Tensor = field(repr=False)
    """(n_probes, n_classes) float32."""


class ProbeMonitor:
    """`update_hook` that scores every returned state on the battery.

    Records one `ProbeReport` per participating client per round. Does nothing
    to the updates and consumes no global randomness.
    """

    def __init__(self, battery: ProbeBattery, device: str = "cpu") -> None:
        """Build the template model that each client state is loaded into.

        It is built under `fork_rng`, so constructing a monitor leaves the
        global RNG untouched.

        Args:
            battery: Battery scored for every update.
            device: Device the template model lives on.
        """
        self.battery = battery
        self.reports: list[ProbeReport] = []
        # build_model seeds the global generator; fork_rng makes that invisible.
        with torch.random.fork_rng(devices=[]):
            self._template = build_model(
                "small_cnn", battery.n_classes, seed=0, device=device
            )

    def score_state(self, state: StateDict) -> torch.Tensor:
        """Score the battery with a model holding `state`.

        Runs under `fork_rng`, so scoring never perturbs the training RNG.

        Args:
            state: Client model state to load into the template.

        Returns:
            (n_probes, n_classes) float32 class probabilities.
        """
        with torch.random.fork_rng(devices=[]):
            self._template.load_state_dict(state)
            return score_probes(self._template, self.battery)

    def __call__(self, updates: list[ClientUpdate], round_idx: int) -> None:
        """Append one `ProbeReport` per update; the `update_hook` entry point.

        Reads the updates and never modifies them.

        Args:
            updates: Client updates returned this round.
            round_idx: Index of the current round.
        """
        for u in updates:
            self.reports.append(
                ProbeReport(
                    client_id=u.client_id,
                    round_idx=round_idx,
                    battery_fingerprint=self.battery.fingerprint,
                    probs=self.score_state(u.state),
                )
            )

    def by_round(self) -> dict[int, list[ProbeReport]]:
        """Return the reports grouped by round index."""
        out: dict[int, list[ProbeReport]] = {}
        for r in self.reports:
            out.setdefault(r.round_idx, []).append(r)
        return out

    def probs_array(self, n_rounds: int, n_clients: int) -> np.ndarray:
        """Stack every report into one array.

        This is the array every statistic in `detect` consumes.

        Args:
            n_rounds: Number of rounds in the run.
            n_clients: Number of clients in the federation.

        Returns:
            (rounds, clients, n_probes, n_classes) float32 probabilities, NaN
            where a client did not participate.
        """
        out = np.full(
            (
                n_rounds,
                n_clients,
                self.battery.n_probes,
                self.battery.n_classes,
            ),
            np.nan,
            dtype=np.float32,
        )
        for r in self.reports:
            out[r.round_idx, r.client_id] = r.probs.numpy()
        return out
