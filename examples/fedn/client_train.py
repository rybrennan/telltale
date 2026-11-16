"""The FEDn server-functions example client, plus a telltale probe report.

Adapted from examples/server-functions/client/train.py in FEDn v0.33.0,
Copyright Scaleout Systems AB, Apache License 2.0. Changes are listed below.

Changes from the upstream example:

  1. After training, the client scores a fixed 50-image probe set with its
     own model and puts the class probabilities in its metadata (1 kB as
     float16, sent as hex because metadata is JSON).
  2. Test rig only: if the server's settings name this site in
     ``telltale_fault``, the client corrupts its own training data from the
     onset round. Fault definitions follow telltale/faults.py.
"""

import hashlib
import math
import os
import re
import sys

import numpy as np
import torch
import torchvision
from fedn.utils.helpers.helpers import load_metadata, save_metadata

from data import load_data
from model import load_parameters, save_parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))

N_PROBES = 50
N_CLASSES = 10
PROBE_SEED = 0


def site_index():
    """1-based data split this client holds, from FEDN_DATA_PATH."""
    m = re.search(
        r"clients/(\d+)/", os.environ.get("FEDN_DATA_PATH", "clients/1/")
    )
    return int(m.group(1))


# -- probes -------------------------------------------------------------------
def probe_battery():
    """The same 50 held-out test images on every client, stratified by class.

    Selection follows telltale.probes.build_probe_battery: one seeded
    permutation per class, read round-robin.
    """
    test = torchvision.datasets.MNIST(root=f"{dir_path}/data/test", train=False)
    y = test.targets.numpy()
    rng = np.random.default_rng([PROBE_SEED, 0x9B0BE])
    per_class = [rng.permutation(np.where(y == k)[0]) for k in range(N_CLASSES)]
    idx = np.array(
        [per_class[i % N_CLASSES][i // N_CLASSES] for i in range(N_PROBES)],
        dtype=np.int64,
    )
    x = (test.data[idx] / 255).contiguous()
    h = hashlib.sha256()
    h.update(idx.tobytes())
    h.update(x.numpy().tobytes())
    return x, h.hexdigest()[:12]


@torch.no_grad()
def score_probes(model, x):
    """Return (n_probes, n_classes) probabilities from one forward pass."""
    model.eval()
    return torch.exp(model(x)).to(torch.float32).numpy()


# -- faults (test rig) --------------------------------------------------------
def magnitude(fault, r):
    if r < fault["onset"]:
        return 0.0
    if fault["ramp"] == 0:
        return float(fault["severity"])
    return float(fault["severity"]) * min(
        1.0, (r - fault["onset"] + 1) / fault["ramp"]
    )


def apply_fault(x, y, fault, r):
    m = magnitude(fault, r)
    if m == 0.0:
        return x, y
    if fault["kind"] == "blur":
        half = 2
        ax = torch.arange(-half, half + 1, dtype=torch.float32)
        k1 = torch.exp(-(ax**2) / (2.0 * m * m))
        k1 = k1 / k1.sum()
        k = torch.outer(k1, k1)
        xb = torch.nn.functional.conv2d(
            x.reshape(-1, 1, 28, 28).float(),
            k.reshape(1, 1, 5, 5),
            padding=half,
        )
        return xb.reshape(x.shape), y
    if fault["kind"] == "label_noise":
        n = int(y.shape[0])
        n_bad = math.floor(m * n)
        if n_bad == 0:
            return x, y
        g = torch.Generator().manual_seed(1000 + site_index())
        order = torch.randperm(n, generator=g)
        offsets = torch.randint(1, N_CLASSES, (n,), generator=g)
        idx = order[:n_bad]
        out = y.clone()
        out[idx] = (y[idx] + offsets[idx]) % N_CLASSES
        return x, out
    raise ValueError(f"unknown fault kind {fault['kind']!r}")


def train(
    in_model_path,
    out_model_path,
    data_path=None,
    batch_size=32,
    epochs=1,
    lr=0.01,
):
    """Complete a model update and report probe scores."""
    # Six clients share one machine in the test rig; one thread each avoids
    # them fighting over cores.
    torch.set_num_threads(1)
    x_train, y_train = load_data(data_path)
    model = load_parameters(in_model_path)

    client_settings = load_metadata(in_model_path)
    lr = client_settings.get("learning_rate", lr)
    r = int(client_settings.get("telltale_round", 0))
    site = site_index()

    fault = client_settings.get("telltale_fault")
    if fault and int(fault["site"]) == site:
        x_train, y_train = apply_fault(x_train, y_train, fault, r)

    # A run seed from the server makes repeated runs differ (sample order
    # and dropout), so healthy runs give a spread to set a threshold from.
    run_seed = int(client_settings.get("telltale_seed", 0))
    torch.manual_seed((run_seed * 7_919 + r) * 1_000_003 + site)
    order = torch.randperm(len(x_train))
    x_train, y_train = x_train[order], y_train[order]
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    n_batches = int(math.ceil(len(x_train) / batch_size))
    criterion = torch.nn.NLLLoss()
    for _ in range(epochs):
        for b in range(n_batches):
            batch_x = x_train[b * batch_size : (b + 1) * batch_size]
            batch_y = y_train[b * batch_size : (b + 1) * batch_size]
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    probes_x, fingerprint = probe_battery()
    probs = score_probes(model, probes_x)

    metadata = {
        "num_examples": len(x_train),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "telltale_probe_scores": probs.astype(np.float16).tobytes().hex(),
        "telltale_shape": [N_PROBES, N_CLASSES],
        "telltale_fingerprint": fingerprint,
        "telltale_site": site,
    }
    save_metadata(metadata, out_model_path)
    save_parameters(model, out_model_path)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
