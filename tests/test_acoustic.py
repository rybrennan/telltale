from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from telltale.acoustic import (
    CLASSES,
    N_FRAMES,
    N_MELS,
    SAMPLE_RATE,
    load_deepship,
    log_mel_image,
    mel_filterbank,
    recording_files,
    segment,
    split_by_recording,
)


def test_mel_filterbank_shape_and_coverage():
    """Filterbank has one non-negative, non-empty row per mel band."""
    fb = mel_filterbank()
    assert fb.shape == (N_MELS, 2048 // 2 + 1)
    assert (fb >= 0).all()
    assert (fb.sum(axis=1) > 0).all()  # every band has support


def test_log_mel_image_is_deterministic_and_shaped():
    """Log-mel image is repeatable, float32, fixed shape, tracks loudness."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=int(2.0 * SAMPLE_RATE))
    a = log_mel_image(x)
    b = log_mel_image(x)
    assert a.shape == (N_MELS, N_FRAMES) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    # A louder signal gives a uniformly higher log-power image.
    assert (log_mel_image(4.0 * x) > a).all()


def test_segment_windows_and_hop():
    """Segmenting yields 2 s windows on a 1 s hop."""
    x = np.zeros(int(5.5 * SAMPLE_RATE))
    wins = segment(x)
    # Starts at 0, 1, 2, 3 s; a 2 s window fits until 3.5 s.
    assert len(wins) == 4
    assert all(len(w) == 2 * SAMPLE_RATE for w in wins)


def _fake_root(tmp_path, per_class=6, seconds=3.0):
    rng = np.random.default_rng(1)
    for k, cls in enumerate(CLASSES):
        d = tmp_path / cls
        d.mkdir()
        for i in range(per_class):
            # Class-specific tone so the classes are separable; int16 to
            # exercise scaling.
            t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
            tone = 0.3 * np.sin(2 * np.pi * (200 + 300 * k) * t)
            x = tone + 0.05 * rng.normal(size=t.size)
            pcm = (x * 32767).astype(np.int16)
            wavfile.write(d / f"{i + 1}.wav", SAMPLE_RATE, pcm)
    return tmp_path


def test_split_is_by_recording_and_disjoint(tmp_path):
    """Split is by whole recording, disjoint, and covers each class."""
    root = _fake_root(tmp_path)
    files = recording_files(root)
    assert len(files) == 24
    train, test = split_by_recording(files, test_every=5)
    # Last recording of each class (6 // 5 = 1).
    assert len(test) == 4 and len(train) == 20
    assert not set(train) & set(test)
    assert {k for _, k in test} == set(range(4))
    # A class with fewer recordings than test_every still gets one test
    # recording.
    dropped = ("3", "4", "5", "6")
    few = [(f, k) for f, k in files if not (k == 3 and f.stem in dropped)]
    tr2, te2 = split_by_recording(few, test_every=5)
    assert (
        sum(1 for _, k in te2 if k == 3) == 1
        and sum(1 for _, k in tr2 if k == 3) == 1
    )


def test_load_deepship_builds_normalised_images_and_caches(tmp_path):
    """Loader builds standardised log-mel images and reloads them from cache."""
    root = _fake_root(tmp_path)
    cache = tmp_path / "cache.pt"
    ds = load_deepship(root, cache=cache)
    assert ds.n_classes == 4 and ds.name == "deepship"
    assert ds.train_x.shape[1:] == (1, N_MELS, N_FRAMES)
    assert ds.train_x.dtype == torch.float32
    # 3 s recordings -> 2 windows each; 20 train recordings -> 40 windows
    assert ds.train_x.shape[0] == 40 and ds.test_x.shape[0] == 8
    assert (
        abs(float(ds.train_x.mean())) < 1e-4
        and abs(float(ds.train_x.std()) - 1.0) < 1e-3
    )
    assert cache.exists()
    again = load_deepship(root, cache=cache)
    torch.testing.assert_close(again.train_x, ds.train_x, rtol=0, atol=0)


def test_missing_root_raises(tmp_path):
    """A missing dataset root raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_deepship(tmp_path / "nope")
