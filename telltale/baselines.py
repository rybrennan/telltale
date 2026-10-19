"""Update-space baselines: what a detector sees if it watches the weights.

telltale watches each client's answers on a probe battery. The detectors in
the literature closest to it watch the model updates instead. This module
records the update-space signals they use, per client per round, and
implements the closest published detector so the two can be scored under the
same protocol on the same runs.

Recorded by `UpdateStatsMonitor` (an `update_hook`):

    norm      ||Δw_k||, the update's length. FedRR's own benchmark signal.
    loo_dist  1 - cos(Δw_k, sample-weighted mean of the other clients' Δw),
              the gradient-space term of FedCE (Jiang et al., CVPR 2023):
              a leave-one-out comparison in update space.
    resid     ||Δw_k - B Bᵀ Δw_k||, the residual off a rank-q subspace B
              fitted to the updates of attack-free start-up rounds. FedRR's
              signal (Xie et al., arXiv:2310.01537).

Scored by `fedrr_cusum`: FedRR's monitoring statistic. Each round the clients
are ranked by the signal, the rank is mapped to a standard normal with a
uniform jitter, and a one-sided CUSUM with allowance d runs per client from
the first monitored round. The paper prints the update as S + Z - d/2, but
its own control limits (K = 5, ARL0 = 30: H = 3.84, 3.28, 2.77 for d = 0.4,
0.5, 0.6) give an in-control ARL near 30 only when the full d is subtracted
(about 11 with d/2), so the full d is used here, matching the chart its
authors ran; tests/test_baselines.py checks this against their limits.
`fedrr_threshold` is FedRR's own control limit: the H that gives a target
in-control average run length under random ranks (their Appendix B), which
is exact only when clients are exchangeable.

Choices that the paper leaves open are fixed here and stated:
    The subspace is fitted to rounds 1-14 (every pre-onset round except round
    0, whose update from the random initialisation would take most of the
    energy). B is the leading left singular vectors of the uncentred update
    matrix, q the fewest that hold 95 percent of its energy; FedRR projects
    Δw onto B without centring. The rank map is Φ⁻¹((rank - U) / K) with the
    largest signal ranked K.

The monitor reads the updates and never modifies them, and consumes no global
randomness, so a run with it attached is bitwise identical to one without.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import norm as _normal

from .fedavg import ClientUpdate, StateDict, aggregate
from .models import build_model

FIT_ROUNDS = range(1, 15)
"""Start-up rounds the FedRR subspace is fitted to. Faults start at 15."""
ENERGY = 0.95
"""Share of the start-up update energy the subspace must hold."""


def flatten(state: StateDict) -> torch.Tensor:
    """Concatenate a state's floating-point tensors into one float64 vector.

    Args:
        state: A model state dict.

    Returns:
        1-D float64 tensor, keys in the state's own order.
    """
    return torch.cat(
        [
            v.detach().reshape(-1).to(torch.float64)
            for v in state.values()
            if torch.is_floating_point(v)
        ]
    )


def fit_subspace(
    deltas: torch.Tensor, energy: float = ENERGY
) -> tuple[torch.Tensor, int, float]:
    """Leading directions of a set of updates, FedRR Phase I.

    Args:
        deltas: (n, p) updates, one per row.
        energy: Share of the total squared norm the subspace must hold.

    Returns:
        (B, q, held): B is (p, q) with orthonormal columns, q the number of
        directions kept, held the share of energy they hold.

    Raises:
        ValueError: If there are no updates or `energy` is outside (0, 1].
    """
    if deltas.ndim != 2 or deltas.shape[0] == 0:
        raise ValueError("need a non-empty (n, p) matrix of updates")
    if not 0.0 < energy <= 1.0:
        raise ValueError("energy must be in (0, 1]")
    u, s, vh = torch.linalg.svd(deltas, full_matrices=False)
    share = torch.cumsum(s**2, 0) / (s**2).sum()
    q = int(torch.searchsorted(share, torch.tensor(energy, dtype=share.dtype)))
    q = min(q + 1, s.numel())
    return vh[:q].T.contiguous(), q, float(share[q - 1])


def residual(delta: torch.Tensor, basis: torch.Tensor) -> float:
    """Length of an update's component outside a subspace.

    Computed as Δw - B(BᵀΔw), never forming BBᵀ.

    Args:
        delta: (p,) update.
        basis: (p, q) orthonormal columns.

    Returns:
        ||Δw - B Bᵀ Δw||.
    """
    return float(torch.linalg.vector_norm(delta - basis @ (basis.T @ delta)))


class UpdateStatsMonitor:
    """`update_hook` recording update-space signals per client per round.

    Keeps the previous global state by re-running the server's own FedAvg
    step on the updates it is shown, so it needs no change to the loop. The
    first global state is rebuilt from the run's init seed exactly as
    `run_federated` builds it.
    """

    def __init__(
        self,
        n_rounds: int,
        n_clients: int,
        n_classes: int,
        init_seed: int,
        fit_rounds: range = FIT_ROUNDS,
        energy: float = ENERGY,
        device: str = "cpu",
    ) -> None:
        """Prepare empty (round, client) arrays and the initial global state.

        Args:
            n_rounds: Rounds in the run.
            n_clients: Clients in the federation.
            n_classes: Output classes of the model.
            init_seed: The run's init seed.
            fit_rounds: Rounds whose updates fit the FedRR subspace.
            energy: Share of start-up energy the subspace holds.
            device: Device the run's model lives on.
        """
        shape = (n_rounds, n_clients)
        self.norm = np.full(shape, np.nan)
        self.norm_preclip = np.full(shape, np.nan)
        self.loo_dist = np.full(shape, np.nan)
        self.resid = np.full(shape, np.nan)
        self.fit_rounds = fit_rounds
        self.energy = energy
        self.q: int | None = None
        self.held: float | None = None
        self._fit: list[torch.Tensor] = []
        self._basis: torch.Tensor | None = None
        with torch.random.fork_rng(devices=[]):
            model = build_model("small_cnn", n_classes, init_seed, device)
        self._global = flatten(model.state_dict())

    def __call__(self, updates: list[ClientUpdate], round_idx: int) -> None:
        """Record this round's signals; the `update_hook` entry point.

        Args:
            updates: Client updates returned this round.
            round_idx: Index of the current round.
        """
        deltas = [flatten(u.sent) - self._global for u in updates]
        weights = np.array([u.n_samples for u in updates], dtype=float)
        total = torch.stack(
            [d * w for d, w in zip(deltas, weights, strict=True)]
        ).sum(0)
        for i, (u, d) in enumerate(zip(updates, deltas, strict=True)):
            k = u.client_id
            self.norm[round_idx, k] = float(torch.linalg.vector_norm(d))
            # What the client trained, before any clipping. Equal to `norm`
            # unless the run clips; a monitor that the client reports to
            # could see this even when the server only sees `sent`.
            self.norm_preclip[round_idx, k] = (
                self.norm[round_idx, k]
                if u.sent_state is None
                else float(
                    torch.linalg.vector_norm(flatten(u.state) - self._global)
                )
            )
            if len(updates) > 1:
                others = (total - d * weights[i]) / (weights.sum() - weights[i])
                cos = torch.nn.functional.cosine_similarity(d, others, dim=0)
                self.loo_dist[round_idx, k] = 1.0 - float(cos)
            if self._basis is not None:
                self.resid[round_idx, k] = residual(d, self._basis)
        if round_idx in self.fit_rounds:
            self._fit.extend(d.to(torch.float32) for d in deltas)
            if round_idx == self.fit_rounds[-1]:
                stack = torch.stack(self._fit).to(torch.float64)
                self._basis, self.q, self.held = fit_subspace(
                    stack, self.energy
                )
                self._fit = []
        self._global = flatten(aggregate(updates))

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the recorded signals and the subspace's size.

        Returns:
            Dict of (round, client) arrays `norm`, `norm_preclip`,
            `loo_dist`, `resid`, plus
            `q`, `held` and `fit_rounds` as arrays for an .npz file.
        """
        return {
            "norm": self.norm,
            "norm_preclip": self.norm_preclip,
            "loo_dist": self.loo_dist,
            "resid": self.resid,
            "q": np.array(-1 if self.q is None else self.q),
            "held": np.array(np.nan if self.held is None else self.held),
            "fit_rounds": np.array(list(self.fit_rounds)),
        }


def local_steps(
    client_sizes: list[int] | np.ndarray, batch_size: int, local_epochs: int
) -> np.ndarray:
    """SGD steps each client takes per round.

    Under one local epoch a client with more data takes more steps and returns
    a longer update, so a statistic that ranks raw update size ranks the
    largest healthy client first every round. Dividing the update-space
    signal by this count is the obvious correction, and the step-normalised
    FedRR variants use it.

    Args:
        client_sizes: Training samples per client.
        batch_size: Local batch size.
        local_epochs: Local epochs per round.

    Returns:
        (clients,) float array of steps per round.
    """
    sizes = np.asarray(client_sizes, dtype=float)
    return local_epochs * np.ceil(sizes / batch_size)


# ── FedRR's monitoring statistic and control limit ────────────────────────────


def rank_scores(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map one round's signal to FedRR's normal scores.

    Clients are ranked 1..K by value, largest K; Z = Φ⁻¹((rank - U) / K) with
    U uniform on (0, 1), so under exchangeable clients Z is exactly standard
    normal. Absent clients (NaN) are left out of the ranking and get NaN.

    Args:
        values: (K,) signal for one round.
        rng: Generator for the uniform jitter.

    Returns:
        (K,) normal scores.
    """
    out = np.full(values.shape, np.nan)
    present = np.flatnonzero(~np.isnan(values))
    if present.size == 0:
        return out
    order = np.argsort(np.argsort(values[present], kind="stable"))
    ranks = order + 1
    u = rng.uniform(size=present.size)
    out[present] = _normal.ppf((ranks - u) / present.size)
    return out


def fedrr_cusum(
    signal: np.ndarray, start: int, d: float = 0.5, seed: int = 0
) -> np.ndarray:
    """FedRR's per-client CUSUM of rank-normal scores.

    S_t = max(0, S_{t-1} + Z_t - d), from round `start` on (the module
    docstring explains why the full d). A client absent
    in a round neither adds nor decays.

    Args:
        signal: (rounds, clients) update-space signal.
        start: First monitored round (T0 + 1 in the paper).
        d: CUSUM reference value; the paper uses 0.4 to 0.6.
        seed: Seed for the rank jitter, so scoring is reproducible.

    Returns:
        (rounds, clients) CUSUM values, NaN before `start` and where absent.
    """
    R, K = signal.shape
    rng = np.random.default_rng(seed)
    out = np.full((R, K), np.nan)
    s = np.zeros(K)
    for r in range(start, R):
        z = rank_scores(signal[r], rng)
        seen = ~np.isnan(z)
        s[seen] = np.maximum(0.0, s[seen] + z[seen] - d)
        out[r, seen] = s[seen]
    return out


def in_control_arl(
    h: float,
    n_clients: int,
    d: float,
    runs: int,
    rng: np.random.Generator,
    cap: int = 100_000,
) -> float:
    """Average run length of FedRR's CUSUM under random ranks.

    FedRR Appendix B, Algorithm 3: ranks are an independent random
    permutation each round; a run ends when any client's S exceeds h.

    Args:
        h: Control limit.
        n_clients: Clients ranked each round.
        d: CUSUM reference value.
        runs: Simulated run lengths to average.
        rng: Generator for permutations and jitter.
        cap: Longest run simulated; a run reaching it counts as `cap`.

    Returns:
        Mean run length in rounds.
    """
    lengths = []
    base = np.arange(1, n_clients + 1)
    for _ in range(runs):
        s = np.zeros(n_clients)
        t = 0
        while t < cap:
            t += 1
            ranks = rng.permutation(base)
            z = _normal.ppf((ranks - rng.uniform(size=n_clients)) / n_clients)
            s = np.maximum(0.0, s + z - d)
            if s.max() > h:
                break
        lengths.append(t)
    return float(np.mean(lengths))


def fedrr_threshold(
    arl0: float,
    n_clients: int,
    d: float = 0.5,
    runs: int = 2000,
    seed: int = 0,
    tol: float = 0.01,
) -> float:
    """FedRR's control limit for a target in-control ARL, by bisection.

    Args:
        arl0: Target in-control average run length (the paper uses 30).
        n_clients: Clients ranked each round.
        d: CUSUM reference value.
        runs: Simulated run lengths per evaluation.
        seed: Seed for the simulation; each evaluation reuses it, so the
            estimated ARL is monotone in h (common random numbers).
        tol: Width of the final bracket on h.

    Returns:
        The control limit h.
    """
    lo, hi = 0.0, 1.0
    while (
        in_control_arl(hi, n_clients, d, runs, np.random.default_rng(seed))
        < arl0
    ):
        lo, hi = hi, hi * 2
    while hi - lo > tol:
        mid = (lo + hi) / 2
        arl = in_control_arl(
            mid, n_clients, d, runs, np.random.default_rng(seed)
        )
        lo, hi = (mid, hi) if arl < arl0 else (lo, mid)
    return hi
