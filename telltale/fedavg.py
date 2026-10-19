"""FedAvg simulation loop.

A direct implementation of McMahan et al. (2017): the server broadcasts a
global model, each selected client trains locally for E epochs, and the server
takes a sample-count-weighted average of the returned parameters.

Runs in-process. There is no networking, no actor scheduling, and no framework
between the loop and the numbers it produces, which is the point. The output
of this harness is a detection probability and a false-alarm rate, and both are
only as trustworthy as our ability to re-run a configuration and get the same
answer.

Everything a downstream integrity detector might want is recorded per client
per round in `ClientUpdate`. Nothing here inspects, scores, or filters those
updates: fault injection and detection are separate layers, added on top,
so that the baseline can be established without them.
"""

from __future__ import annotations

import copy
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from .config import FedConfig
from .data import Dataset
from .models import build_model, count_parameters

StateDict = dict[str, torch.Tensor]
ShardTransform = Callable[
    [int, int, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


# ── determinism ───────────────────────────────────────────────────────────────


def seed_everything(seed: int) -> None:
    """Seed the Python, NumPy and torch global generators.

    NumPy's legacy seed must be below 2**32, hence the modulus.

    Args:
        seed: Seed for all three generators.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def _derive_seed(*parts: int) -> int:
    """Stable seed from a tuple of ints, e.g. (train_seed, round, client).

    Deriving per-client seeds this way means a client trains identically
    regardless of how many other clients ran before it in the loop, so results
    do not depend on iteration order.

    Args:
        *parts: Integers to combine, in order.

    Returns:
        A seed in [0, 2**31 - 1).
    """
    h = 0
    for p in parts:
        h = (h * 1_000_003 + int(p)) % (2**31 - 1)
    return h


# ── records ───────────────────────────────────────────────────────────────────


@dataclass
class ClientUpdate:
    """One client's contribution in one round."""

    client_id: int
    round_idx: int
    n_samples: int
    train_loss: float
    state: StateDict = field(repr=False)
    sent_state: StateDict | None = field(default=None, repr=False)
    """What the client sends for aggregation when it differs from `state`
    (a clipped update). None means it sends `state` itself."""

    @property
    def sent(self) -> StateDict:
        """The state the server aggregates."""
        return self.state if self.sent_state is None else self.sent_state


@dataclass
class RoundRecord:
    """One round's outcome.

    Who reported, the global model's test metrics, the mean client training
    loss, and wall-clock seconds.
    """

    round_idx: int
    participants: list[int]
    test_loss: float
    test_acc: float
    mean_train_loss: float
    seconds: float

    def to_dict(self) -> dict:
        """Return the JSON form.

        Returns:
            A dict of the record's fields. The round index is keyed `round`,
            not `round_idx`.
        """
        return {
            "round": self.round_idx,
            "participants": self.participants,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "mean_train_loss": self.mean_train_loss,
            "seconds": self.seconds,
        }


@dataclass
class FedResult:
    """A finished run.

    Holds the config, per-round records, parameter count, the partition's skew
    summary, and the final global state.
    """

    config: FedConfig
    rounds: list[RoundRecord]
    n_parameters: int
    partition_summary: dict
    final_state: StateDict = field(repr=False)

    @property
    def final_acc(self) -> float:
        """Test accuracy after the last round."""
        return self.rounds[-1].test_acc

    def accuracy_curve(self) -> list[float]:
        """Return test accuracy per round.

        Returns:
            Test accuracy of each round, in round order.
        """
        return [r.test_acc for r in self.rounds]

    def to_dict(self) -> dict:
        """Return a JSON-serialisable summary.

        Includes the config fingerprint so the file traces to its run. Omits
        `final_state`.

        Returns:
            A dict with keys `config`, `fingerprint`, `n_parameters`,
            `partition_summary`, `rounds` and `final_acc`.
        """
        return {
            "config": self.config.to_dict(),
            "fingerprint": self.config.fingerprint(),
            "n_parameters": self.n_parameters,
            "partition_summary": self.partition_summary,
            "rounds": [r.to_dict() for r in self.rounds],
            "final_acc": self.final_acc,
        }


# ── local training ────────────────────────────────────────────────────────────


def local_train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: FedConfig,
    seed: int,
) -> tuple[StateDict, float]:
    """Train `model` in place on one client's shard.

    Batching is done by hand from a seeded permutation rather than through a
    DataLoader so that batch composition is a pure function of `seed`.

    Args:
        model: Model to train; modified in place.
        x: The client's inputs.
        y: The client's labels.
        cfg: Run config; supplies epochs, batch size, lr and momentum.
        seed: Seed for the batch-order generator.

    Returns:
        A tuple (state, mean loss): a detached copy of the trained state dict
        and the mean training loss over all batches.
    """
    model.train()
    optimiser = torch.optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum
    )
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    n = x.shape[0]
    total_loss, n_batches = 0.0, 0

    for _ in range(cfg.local_epochs):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            if idx.numel() == 0:
                continue
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x[idx]), y[idx])
            loss.backward()
            optimiser.step()
            total_loss += float(loss.detach())
            n_batches += 1

    return (
        {k: v.detach().clone() for k, v in model.state_dict().items()},
        total_loss / max(n_batches, 1),
    )


# ── aggregation ───────────────────────────────────────────────────────────────


def aggregate(updates: Iterable[ClientUpdate]) -> StateDict:
    """Average client parameters weighted by sample count.

    This is the FedAvg server step. Floating-point tensors are accumulated in
    float64 and cast back to their original dtype.

    Args:
        updates: Client updates for one round.

    Returns:
        The aggregated state dict.

    Raises:
        ValueError: If there are no updates, or their sample counts sum to
            zero or less.
    """
    updates = list(updates)
    if not updates:
        raise ValueError("no client updates to aggregate")

    total = sum(u.n_samples for u in updates)
    if total <= 0:
        raise ValueError("client sample counts sum to zero")

    out: StateDict = {}
    for key, ref in updates[0].sent.items():
        if not torch.is_floating_point(ref):
            # Integer buffers (none in SmallCNN, but guard the general case):
            # averaging them is meaningless, so carry the first client's value.
            out[key] = ref.clone()
            continue
        acc = torch.zeros_like(ref, dtype=torch.float64)
        for u in updates:
            acc += u.sent[key].to(torch.float64) * (u.n_samples / total)
        out[key] = acc.to(ref.dtype)
    return out


# ── evaluation ────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 512
) -> tuple[float, float]:
    """Evaluate `model` on a labelled set.

    Args:
        model: Model to evaluate; switched to eval mode.
        x: Inputs.
        y: Labels.
        batch_size: Samples per forward pass.

    Returns:
        A tuple (mean loss, accuracy).
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct = 0.0, 0
    for start in range(0, x.shape[0], batch_size):
        xb, yb = x[start : start + batch_size], y[start : start + batch_size]
        logits = model(xb)
        total_loss += float(criterion(logits, yb))
        correct += int((logits.argmax(dim=1) == yb).sum())
    n = x.shape[0]
    return total_loss / n, correct / n


# ── the loop ──────────────────────────────────────────────────────────────────


def clip_update(
    state: StateDict, global_state: StateDict, clip_norm: float
) -> StateDict:
    """Scale a client's update down to at most `clip_norm` in L2 norm.

    The update is `state - global_state` over floating-point tensors; it is
    multiplied by min(1, clip_norm / ||update||) and added back to the global
    state, as DP-FedAvg does before aggregation. Integer buffers are copied.

    Args:
        state: The client's trained state.
        global_state: The global state it trained from.
        clip_norm: Largest L2 norm the sent update may have.

    Returns:
        The state the client sends.
    """
    keys = [k for k, v in state.items() if torch.is_floating_point(v)]
    sq = sum(
        float(((state[k].double() - global_state[k].double()) ** 2).sum())
        for k in keys
    )
    norm = sq**0.5
    if norm <= clip_norm:
        return {k: v.clone() for k, v in state.items()}
    scale = clip_norm / norm
    out: StateDict = {}
    for k, v in state.items():
        if k in keys:
            g = global_state[k]
            out[k] = (g.double() + (v.double() - g.double()) * scale).to(
                v.dtype
            )
        else:
            out[k] = v.clone()
    return out


def _select_participants(
    rnd: int,
    cfg: FedConfig,
    participation: Callable[[int], list[int]] | None,
    selection_rng: np.random.Generator,
) -> list[int]:
    """Sorted client ids that report in round `rnd`.

    An explicit `participation` schedule wins; otherwise every client reports,
    or a `client_fraction` sample is drawn. `selection_rng` is only advanced on
    the sampling branch, so a schedule or full participation never perturbs it.

    Raises:
        ValueError: If the schedule returns no clients or an id outside
            [0, n_clients).
    """
    if participation is not None:
        participants = sorted({int(c) for c in participation(rnd)})
        if (
            not participants
            or participants[0] < 0
            or participants[-1] >= cfg.n_clients
        ):
            raise ValueError(
                f"participation returned {participants} "
                f"for {cfg.n_clients} clients"
            )
        return participants
    if cfg.clients_per_round >= cfg.n_clients:
        return list(range(cfg.n_clients))
    return sorted(
        selection_rng.choice(
            cfg.n_clients, size=cfg.clients_per_round, replace=False
        ).tolist()
    )


def _client_update(
    model: nn.Module,
    global_state: StateDict,
    shard: tuple[torch.Tensor, torch.Tensor],
    cid: int,
    rnd: int,
    cfg: FedConfig,
    fault: ShardTransform | None,
) -> ClientUpdate:
    """Run one client's round.

    Applies any fault to the client's shard, then trains from the global
    model.
    """
    x, y = shard
    if fault is not None:
        x, y = fault(cid, rnd, x, y)
    local = copy.deepcopy(model)
    local.load_state_dict(global_state)
    state, loss = local_train(
        local, x, y, cfg, seed=_derive_seed(cfg.train_seed, rnd, cid)
    )
    return ClientUpdate(
        client_id=cid,
        round_idx=rnd,
        n_samples=int(x.shape[0]),
        train_loss=loss,
        state=state,
    )


def run_federated(
    cfg: FedConfig,
    dataset: Dataset,
    parts: list[np.ndarray],
    partition_summary: dict | None = None,
    on_round: Callable[[RoundRecord], None] | None = None,
    update_hook: Callable[[list[ClientUpdate], int], None] | None = None,
    fault: ShardTransform | None = None,
    participation: Callable[[int], list[int]] | None = None,
) -> FedResult:
    """Run a full federation and return the per-round history.

    `update_hook` receives every client's update before aggregation each round.
    It is the seam the integrity layer attaches to; the baseline passes nothing
    and the loop behaves identically with or without it.

    `fault` is the second seam: called as fault(client_id, round_idx, x, y) on
    each client's shard before local training, returning the (x, y) the client
    actually trains on. It is how fault injection reaches a node's feed without
    the federation code knowing anything about faults. None means every client
    trains on its true shard, and a fault that returns its inputs untouched is
    bitwise indistinguishable from None.

    `participation` is the third seam: called as participation(round_idx) and
    returning the client ids that report this round. It is how a contact
    schedule (submarines surfacing every k rounds) reaches the loop. None
    means every client (or the `client_fraction` sample). A schedule that
    returns every client every round is bitwise indistinguishable from None.

    Args:
        cfg: Run config.
        dataset: The data; client shards are sliced from its training split.
        parts: One index array per client, as from `dirichlet_partition`.
        partition_summary: Skew summary stored on the result; None stores
            an empty dict.
        on_round: Called with each `RoundRecord` after the round finishes.
        update_hook: Called as update_hook(updates, round_idx) before
            aggregation.
        fault: Shard transform applied before local training.
        participation: Per-round schedule of reporting client ids.

    Returns:
        The finished run.

    Raises:
        ValueError: If `parts` does not have one entry per client, or a
            participation schedule returns an invalid client set.
    """
    if len(parts) != cfg.n_clients:
        raise ValueError(
            f"got {len(parts)} partitions for {cfg.n_clients} clients"
        )

    seed_everything(cfg.init_seed)
    model = build_model(
        "small_cnn", dataset.n_classes, cfg.init_seed, cfg.device
    )
    global_state = {
        k: v.detach().clone() for k, v in model.state_dict().items()
    }
    n_params = count_parameters(model)

    # Pre-slice each client's shard once rather than indexing every round.
    shards = [
        (
            dataset.train_x[idx].to(cfg.device),
            dataset.train_y[idx].to(cfg.device),
        )
        for idx in parts
    ]
    test_x = dataset.test_x.to(cfg.device)
    test_y = dataset.test_y.to(cfg.device)

    selection_rng = np.random.default_rng([cfg.train_seed, 0xC11E])
    history: list[RoundRecord] = []

    for rnd in range(cfg.rounds):
        started = time.perf_counter()

        participants = _select_participants(
            rnd, cfg, participation, selection_rng
        )
        updates = [
            _client_update(
                model, global_state, shards[cid], cid, rnd, cfg, fault
            )
            for cid in participants
        ]

        if cfg.clip_norm is not None:
            for u in updates:
                u.sent_state = clip_update(u.state, global_state, cfg.clip_norm)

        if update_hook is not None:
            update_hook(updates, rnd)

        global_state = aggregate(updates)
        model.load_state_dict(global_state)
        test_loss, test_acc = evaluate(
            model, test_x, test_y, cfg.eval_batch_size
        )

        record = RoundRecord(
            round_idx=rnd,
            participants=participants,
            test_loss=test_loss,
            test_acc=test_acc,
            mean_train_loss=float(np.mean([u.train_loss for u in updates])),
            seconds=time.perf_counter() - started,
        )
        history.append(record)
        if on_round is not None:
            on_round(record)

    return FedResult(
        config=cfg,
        rounds=history,
        n_parameters=n_params,
        partition_summary=partition_summary or {},
        final_state=global_state,
    )
