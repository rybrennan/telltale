"""Fault injection with known ground truth.

A fault is a transform on one node's training shard: which node, which round it
starts, how fast it ramps, what kind, how severe. Everything is known by
construction, so a detection later is a detection of *this* fault at *this*
node from *this* round, never an inference from labels.

The case the harness exists for is the slow ramp: a magnitude that rises
linearly from zero over `ramp` rounds, so that at no single round is the
change gross. Step faults (`ramp=0`) are the easy control.

Faults reach the loop through the `fault` seam of `run_federated`. Before onset
the transform returns its inputs untouched (the same tensor objects), so a
fault configured but not yet active is bitwise indistinguishable from no fault.
The same holds for severity zero. A test asserts both.

Transforms act in the normalised input space the model sees, where 1.0 is one
standard deviation of pixel intensity:

  bias         x + m               additive offset (sensor DC drift)
  gain         x * (1 + m)         contrast change about the channel mean
  blur         gaussian blur, sigma = m pixels (optics degrading)
  label_noise  fraction m of the shard relabelled uniformly at random; the
               corrupted set is nested as m grows, so a ramp is monotone

Label noise is the "miscalibrated ground truth at the node" case; the other
three are "the feed itself is drifting inside nominal bounds".
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .fedavg import ShardTransform

KINDS = ("bias", "gain", "blur", "label_noise")


@dataclass(frozen=True)
class FaultSpec:
    """One fault on one node.

    Holds its kind, first active round, severity and ramp length. Frozen and
    fingerprintable, so a result names the fault that produced it.
    """

    kind: str
    node: int
    onset: int
    """First round (0-based) at which the fault is active."""
    severity: float
    """Magnitude reached at the end of the ramp, in the units listed above."""
    ramp: int = 0
    """Rounds over which the magnitude rises linearly to `severity`.

    0 means a step.
    """
    seed: int = 0
    """Only label_noise draws randomness; derived per (seed, node)."""

    def __post_init__(self) -> None:
        """Validate the spec.

        Raises:
            ValueError: If the kind is unknown, the node, onset, ramp or
                severity is negative, or a label_noise severity is above 1
                (it is a fraction).
        """
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.node < 0:
            raise ValueError("node must be >= 0")
        if self.onset < 0:
            raise ValueError("onset must be >= 0")
        if self.ramp < 0:
            raise ValueError("ramp must be >= 0")
        if self.severity < 0:
            raise ValueError("severity must be >= 0")
        if self.kind == "label_noise" and self.severity > 1.0:
            raise ValueError("label_noise severity is a fraction in [0, 1]")

    def magnitude(self, round_idx: int) -> float:
        """Return the severity reached at `round_idx`.

        0 before onset, linear over the ramp, `severity` from onset + ramp
        onward. A step reaches full severity at the onset round itself.

        Args:
            round_idx: 0-based round index.

        Returns:
            The fault magnitude in that round.
        """
        if round_idx < self.onset:
            return 0.0
        if self.ramp == 0:
            return self.severity
        return self.severity * min(
            1.0, (round_idx - self.onset + 1) / self.ramp
        )

    def to_dict(self) -> dict:
        """Return the plain-dict form.

        Returns:
            The dataclass fields as a dict; the input to `fingerprint`.
        """
        return asdict(self)

    def fingerprint(self) -> str:
        """Return a stable hash of the spec.

        Returns:
            The first 12 hex digits of the SHA-256 of the sorted-key JSON.
        """
        blob = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ── transforms ────────────────────────────────────────────────────────────────


def _gaussian_kernel(sigma: float, size: int = 5) -> torch.Tensor:
    """Build a normalised (size, size) Gaussian kernel from a 1-D one."""
    half = size // 2
    ax = torch.arange(-half, half + 1, dtype=torch.float32)
    k1 = torch.exp(-(ax**2) / (2.0 * sigma * sigma))
    k1 = k1 / k1.sum()
    return torch.outer(k1, k1)


def transform_inputs(x: torch.Tensor, kind: str, m: float) -> torch.Tensor:
    """Apply an input-space fault of magnitude `m`.

    Never mutates `x`.

    Args:
        x: Input batch, shape (n, channels, height, width).
        kind: One of "bias", "gain" or "blur".
        m: Fault magnitude.

    Returns:
        A new tensor, or `x` itself (same object) at m == 0.

    Raises:
        ValueError: If `kind` is not an input-space fault.
    """
    if m == 0.0:
        return x
    if kind == "bias":
        return x + m
    if kind == "gain":
        return x * (1.0 + m)
    if kind == "blur":
        k = _gaussian_kernel(m).to(x.dtype).to(x.device)
        c = x.shape[1]
        weight = k.expand(c, 1, *k.shape).contiguous()
        return F.conv2d(x, weight, padding=k.shape[-1] // 2, groups=c)
    raise ValueError(f"{kind!r} is not an input-space fault")


def corrupt_labels(
    y: torch.Tensor, m: float, n_classes: int, seed: int
) -> torch.Tensor:
    """Relabel floor(m * n) samples uniformly at random to a *different* class.

    The order of corruption is a fixed permutation drawn from `seed`, so the set
    corrupted at magnitude m is a subset of the set corrupted at any m' > m.
    That is what makes a ramp monotone: labels corrupted in round t stay
    corrupted in round t + 1.

    Args:
        y: Labels of the shard.
        m: Fraction of the shard to relabel.
        n_classes: Number of classes.
        seed: Seed for the corruption order and offsets.

    Returns:
        A relabelled copy, or `y` itself (same object) when no sample is
        relabelled.
    """
    n = int(y.shape[0])
    k = math.floor(m * n)
    if k == 0:
        return y
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g)
    # A fixed random offset in 1..n_classes-1 per sample guarantees a change.
    offsets = torch.randint(1, n_classes, (n,), generator=g)
    idx = order[:k]
    out = y.clone()
    out[idx] = (y[idx] + offsets[idx]) % n_classes
    return out


# ── the seam ──────────────────────────────────────────────────────────────────


def _derive_seed(*parts: int) -> int:
    """Derive a stable seed from a tuple of ints.

    Same construction as `fedavg._derive_seed`.
    """
    h = 0
    for p in parts:
        h = (h * 1_000_003 + int(p)) % (2**31 - 1)
    return h


def make_fault(specs: list[FaultSpec], n_classes: int) -> ShardTransform:
    """Build the `fault` callable for `run_federated` from one or more specs.

    Specs on different nodes compose; two specs on the same node apply in
    list order. Nodes with no spec, and rounds before onset, pass through
    untouched.

    Args:
        specs: Fault specs to apply.
        n_classes: Number of classes, needed by label noise.

    Returns:
        A callable (client_id, round_idx, x, y) -> (x, y).
    """
    by_node: dict[int, list[FaultSpec]] = {}
    for s in specs:
        by_node.setdefault(s.node, []).append(s)

    def fault(
        client_id: int, round_idx: int, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply every spec on `client_id` that is active in `round_idx`.

        Label noise rewrites `y`; every other kind transforms `x`.

        Args:
            client_id: Node whose shard is being transformed.
            round_idx: Current round.
            x: Shard inputs.
            y: Shard labels.

        Returns:
            (x, y), the same objects when nothing is active.
        """
        for s in by_node.get(client_id, ()):
            m = s.magnitude(round_idx)
            if m == 0.0:
                continue
            if s.kind == "label_noise":
                y = corrupt_labels(
                    y, m, n_classes, _derive_seed(s.seed, s.node)
                )
            else:
                x = transform_inputs(x, s.kind, m)
        return x, y

    return fault
