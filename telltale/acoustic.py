"""Underwater acoustic modality: DeepShip recordings as log-mel images.

DeepShip hydrophone recordings become 28x28 log-mel spectrogram images.

Why a second modality: the stage-4 results are on clothing photographs, and
the objection "that is not sonar" is fair. DeepShip (Irfan et al., 2021) is
real underwater ship-radiated noise recorded in the Strait of Georgia, four
vessel classes (cargo, passenger ship, tanker, tug), used across the academic
underwater acoustic target recognition literature. The public portion on
GitHub is about ninety minutes of audio; it is used unmodified.

The whole point of the harness is that the instrument does not care what the
input is. So the audio is turned into the same shape the rest of the code
already handles, (N, 1, 28, 28) float32, normalised, and nothing downstream
changes: same model, same probe battery, same faults, same statistics. The
faults keep their physical meaning in a time-frequency image: bias is a DC
offset in log power (receiver gain drift), gain is contrast (dynamic-range
compression), blur is time-frequency smearing (bandwidth loss), label noise is
a mislabelled contact.

Segmentation and leakage: recordings are cut into 2-second windows with a
1-second hop. The train/test split is BY RECORDING, never by window, because
adjacent windows of one recording are near-duplicates and a window-level split
would leak the test set into training. The last fifth of each class's
recordings (by sorted filename), and at least one, is held out.

This is a small dataset for a 200k-parameter CNN, and a synthetic unit-test
fixture verifies optimisation, not generalisation; the same caution applies
here at a larger scale. The claims made on it are about the instrument's
behaviour under injected faults, which is what the harness measures, not about
vessel-classification accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile

from .data import Dataset

CLASSES = ("Cargo", "Passengership", "Tanker", "Tug")
SAMPLE_RATE = 32_000
WINDOW_S, HOP_S = 2.0, 1.0
N_FFT = 2048
N_MELS = 28
N_FRAMES = 28
F_MIN, F_MAX = 20.0, 8000.0
TEST_EVERY = 5


def _hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    """Convert Hz to the HTK mel scale: 2595 * log10(1 + f / 700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    """Convert HTK mel back to Hz; the inverse of `_hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    f_min: float = F_MIN,
    f_max: float = F_MAX,
) -> np.ndarray:
    """Build triangular filters on a mel-spaced grid.

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT length in samples.
        n_mels: Number of mel bands.
        f_min: Lowest band edge in Hz.
        f_max: Highest band edge in Hz.

    Returns:
        (n_mels, n_fft // 2 + 1) float64 filter weights.
    """
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    edges = _mel_to_hz(
        np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    )
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(mid - lo, 1e-9)
        down = (hi - freqs) / max(hi - mid, 1e-9)
        fb[m] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


def log_mel_image(
    x: np.ndarray, sr: int = SAMPLE_RATE, fb: np.ndarray | None = None
) -> np.ndarray:
    """Turn one window of audio into a log-mel power image.

    STFT hop is chosen so the window yields exactly N_FRAMES frames; the
    result is deterministic for a given input.

    Args:
        x: One window of mono audio samples.
        sr: Sample rate in Hz; used only when `fb` is None.
        fb: Precomputed mel filterbank, or None to build one for `sr`.

    Returns:
        (N_MELS, N_FRAMES) float32 natural-log mel power.
    """
    fb = mel_filterbank(sr) if fb is None else fb
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, (len(x) - N_FFT) // (N_FRAMES - 1))
    win = np.hanning(N_FFT)
    frames = np.stack(
        [x[i * hop : i * hop + N_FFT] * win for i in range(N_FRAMES)]
    )
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # (frames, bins)
    mel = fb @ power.T  # (mels, frames)
    return np.log(mel + 1e-10).astype(np.float32)


def segment(x: np.ndarray, sr: int = SAMPLE_RATE) -> list[np.ndarray]:
    """Cut `x` into WINDOW_S-second windows every HOP_S seconds.

    A trailing partial window is dropped, so a recording shorter than one
    window yields none.

    Args:
        x: Mono audio samples.
        sr: Sample rate in Hz.

    Returns:
        The windows, as slices of `x`.
    """
    n_win, n_hop = int(WINDOW_S * sr), int(HOP_S * sr)
    return [x[s : s + n_win] for s in range(0, len(x) - n_win + 1, n_hop)]


def recording_files(root: Path) -> list[tuple[Path, int]]:
    """List every wav, sorted by class then filename.

    Filenames sort by (length, stem), so numeric stems order naturally.

    Args:
        root: Directory holding one subdirectory per class in CLASSES.

    Returns:
        (path, class index) pairs.
    """
    out = []
    for k, cls in enumerate(CLASSES):
        wavs = (root / cls).glob("*.wav")
        for f in sorted(wavs, key=lambda p: (len(p.stem), p.stem)):
            out.append((f, k))
    return out


def split_by_recording(
    files: list[tuple[Path, int]], test_every: int = TEST_EVERY
) -> tuple[list, list]:
    """Split recordings into train and test, per class.

    Per class, the LAST max(1, n // test_every) recordings by sorted name
    are test. At least one per class, because the probe battery is drawn from
    the test split and must cover every class; the public DeepShip portion has
    only four tug recordings.

    Args:
        files: (path, class index) pairs from `recording_files`.
        test_every: One in this many recordings per class is held out.

    Returns:
        (train, test) lists of (path, class index) pairs.
    """
    by_class: dict[int, list] = {}
    for f, k in files:
        by_class.setdefault(k, []).append((f, k))
    train, test = [], []
    for k in sorted(by_class):
        items = by_class[k]
        n_test = max(1, len(items) // test_every)
        train += items[:-n_test]
        test += items[-n_test:]
    return train, test


def _standardise(images: np.ndarray, mean: float, std: float) -> torch.Tensor:
    """Standardise (N, 28, 28) images into an (N, 1, 28, 28) tensor."""
    return torch.from_numpy((images - mean) / std).unsqueeze(1).contiguous()


def load_deepship(root: Path, cache: Path | None = None) -> Dataset:
    """Build (or load from cache) the DeepShip dataset as 28x28 images.

    Normalisation uses the TRAIN split's mean and SD, applied to both splits,
    so the test split does not inform the scale.

    Args:
        root: Directory holding one subdirectory of wavs per class.
        cache: Optional path of a cached tensor blob; read if it exists,
            written after a build otherwise.

    Returns:
        The dataset with standardised train and test splits.

    Raises:
        FileNotFoundError: If no wav files are found under `root`.
    """
    root = Path(root)
    if cache is not None and Path(cache).exists():
        blob = torch.load(cache)
        return Dataset(**blob, n_classes=len(CLASSES), name="deepship")

    files = recording_files(root)
    if not files:
        raise FileNotFoundError(f"no DeepShip wav files under {root}")
    train_files, test_files = split_by_recording(files)
    fb = mel_filterbank()

    def build(
        items: list[tuple[Path, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn (file, label) pairs into un-normalised log-mel images.

        Multi-channel audio is averaged to mono and integer PCM is scaled to
        [-1, 1]. The caller normalises with the TRAIN split's statistics.

        Args:
            items: (path, class index) pairs.

        Returns:
            (images, labels): stacked log-mel images and int64 labels.
        """
        xs, ys = [], []
        for f, k in items:
            sr, x = wavfile.read(f)
            if x.ndim > 1:
                x = x.mean(axis=1)
            if x.dtype.kind in "iu":
                x = x.astype(np.float64) / np.iinfo(x.dtype).max
            for w in segment(x, sr):
                xs.append(log_mel_image(w, sr, fb))
                ys.append(k)
        return np.stack(xs), np.asarray(ys, dtype=np.int64)

    xtr, ytr = build(train_files)
    xte, yte = build(test_files)
    mean, std = float(xtr.mean()), float(xtr.std())
    blob = {
        "train_x": _standardise(xtr, mean, std),
        "train_y": torch.from_numpy(ytr),
        "test_x": _standardise(xte, mean, std),
        "test_y": torch.from_numpy(yte),
    }
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, cache)
    return Dataset(**blob, n_classes=len(CLASSES), name="deepship")
