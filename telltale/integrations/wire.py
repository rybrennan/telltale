"""The probe-score wire format shared by every framework adapter.

A client sends (n_probes, n_classes) class probabilities as float16 bytes. At
50 probes and 10 classes that is 1,000 bytes per client per round.
"""

from __future__ import annotations

import numpy as np


def encode_scores(probs: np.ndarray) -> bytes:
    """Encode probabilities as float16 bytes; this is the payload.

    Args:
        probs: (n_probes, n_classes) class probabilities.

    Returns:
        The raw float16 bytes in C order.
    """
    return np.asarray(probs, dtype=np.float16).tobytes()


def decode_scores(blob: bytes, n_probes: int, n_classes: int) -> np.ndarray:
    """Decode float16 bytes; the inverse of `encode_scores`.

    Values come back at float16 precision; that is the wire format.

    Args:
        blob: Payload produced by `encode_scores`.
        n_probes: Number of probes in the battery.
        n_classes: Number of classes per row.

    Returns:
        (n_probes, n_classes) float32 probabilities.
    """
    return (
        np.frombuffer(blob, dtype=np.float16)
        .reshape(n_probes, n_classes)
        .astype(np.float32)
    )
