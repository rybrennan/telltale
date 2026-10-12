"""Divergence statistics over probe reports.

Every function here consumes probability arrays and returns numbers. Nothing
here touches the model, the data, or the training loop. The array convention
throughout is

    probs[round, client, probe, class]      float, NaN for an absent client

as produced by `ProbeMonitor.probs_array`.

Three statistics, one per handoff bullet, plus the instrument self-check:

  level        mean JS divergence of a node's rows from its peers' leave-one-out
               consensus, z-scored within the round against that round's own
               median and MAD (the locally measured floor)
  self         mean JS divergence of a node's rows from its own rows last round;
               needs no peers
  persistence  whether the change in a node's offset-from-consensus, measured
               against its own trailing mean, is directional across rounds

No fault exists at this stage, so nothing here is a detector yet. These are
the quantities whose healthy distribution defines the false-alarm side of the
operating curve; the detection side is measured in stage 4 against stage 3's
faults. See docs/stage2_probe_design.md for why *level* alone cannot be the
detector.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import numpy as np
import torch

from .probes import ProbeBattery, ProbeReport, score_probes

EPS = 1e-12
LN2 = float(np.log(2.0))
MAD_TO_SD = 1.4826


# ── divergences (row-wise over the last axis) ─────────────────────────────────


def _clip(p: np.ndarray) -> np.ndarray:
    """Cast to float64 and clip to [EPS, 1] so log() is finite on zeros."""
    return np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0)


def kl(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) in nats, per row.

    Args:
        p: Probability rows over the last axis.
        q: Probability rows over the last axis, broadcastable with `p`.

    Returns:
        KL divergence per row.
    """
    p, q = _clip(p), _clip(q)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence in nats, per row.

    Symmetric, bounded by ln 2.

    Args:
        p: Probability rows over the last axis.
        q: Probability rows over the last axis, broadcastable with `p`.

    Returns:
        JS divergence per row.
    """
    p, q = _clip(p), _clip(q)
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ── cross-node level ──────────────────────────────────────────────────────────


def loo_consensus(probs_round: np.ndarray) -> np.ndarray:
    """Leave-one-out peer mean, (clients, probes, classes).

    Row k is the mean over every *other* present client. A client is never
    compared against a consensus it contributed to. An absent client is
    compared against the mean of every present client.

    Args:
        probs_round: (clients, probes, classes) probabilities for one round,
            NaN for an absent client.

    Returns:
        (clients, probes, classes) consensus, NaN where fewer than one peer
        is present.
    """
    p = np.asarray(probs_round, dtype=np.float64)
    present = ~np.isnan(p).any(axis=(1, 2))
    total = np.nansum(p[present], axis=0)
    n_present = int(present.sum())
    out = np.full_like(p, np.nan)
    for k in range(p.shape[0]):
        if present[k] and n_present >= 2:
            out[k] = (total - p[k]) / (n_present - 1)
        elif not present[k] and n_present >= 1:
            out[k] = total / n_present
    return out


def level_divergence(probs_round: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-client mean JS to the LOO consensus, and the offset vectors.

    Args:
        probs_round: (clients, probes, classes) probabilities for one round.

    Returns:
        (level[clients], offset[clients, probes, classes]), NaN for a client
        that is absent or has no consensus.
    """
    p = np.asarray(probs_round, dtype=np.float64)
    q = loo_consensus(p)
    level = np.full(p.shape[0], np.nan)
    offset = np.full_like(p, np.nan)
    for k in range(p.shape[0]):
        if not (np.isnan(p[k]).any() or np.isnan(q[k]).any()):
            level[k] = js(p[k], q[k]).mean()
            offset[k] = p[k] - q[k]
    return level, offset


def robust_z(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-scores against the median and scaled MAD of `values`, ignoring NaN.

    If the MAD is zero the scale falls back to the standard deviation; if
    that is zero too every z is 0: nothing can be outlying in a set of
    identical numbers.

    Args:
        values: 1-D values, NaN where missing.

    Returns:
        (z, median, scale). All NaN if fewer than two values are finite.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    if ok.sum() < 2:
        return np.full_like(v, np.nan), float("nan"), float("nan")
    med = float(np.median(v[ok]))
    scale = MAD_TO_SD * float(np.median(np.abs(v[ok] - med)))
    if scale <= 0.0:
        scale = float(np.std(v[ok]))
    z = np.full_like(v, np.nan)
    z[ok] = 0.0 if scale <= 0.0 else (v[ok] - med) / scale
    return z, med, scale


def level_series(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Level statistics for every round.

    Args:
        probs: (rounds, clients, probes, classes) probabilities.

    Returns:
        Dict with

          level   (rounds, clients)          mean JS to LOO consensus
          z       (rounds, clients)          within-round robust z
          offset  (rounds, clients, probes, classes)
          median  (rounds,)                  within-round median level
          scale   (rounds,)                  within-round robust scale
    """
    R, K = probs.shape[:2]
    level = np.full((R, K), np.nan)
    z = np.full((R, K), np.nan)
    offset = np.full(probs.shape, np.nan)
    median = np.full(R, np.nan)
    scale = np.full(R, np.nan)
    for r in range(R):
        level[r], offset[r] = level_divergence(probs[r])
        z[r], median[r], scale[r] = robust_z(level[r])
    return {
        "level": level,
        "z": z,
        "offset": offset,
        "median": median,
        "scale": scale,
    }


# ── temporal self ─────────────────────────────────────────────────────────────


def _observed_rounds(a: np.ndarray, k: int) -> np.ndarray:
    """Round indices where client k's entry is fully finite."""
    x = a[:, k].reshape(a.shape[0], -1)
    return np.nonzero(~np.isnan(x).any(axis=1))[0]


def self_divergence(probs: np.ndarray) -> np.ndarray:
    """Mean JS between a node's rows and its rows at its previous contact.

    The previous contact is the last round the node reported, not
    necessarily the round before.

    Args:
        probs: (rounds, clients, probes, classes) probabilities.

    Returns:
        (rounds, clients) self-divergence, NaN at a node's first contact and
        wherever it is absent.
    """
    R, K = probs.shape[:2]
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(probs, k)
        for i in range(1, len(obs)):
            out[obs[i], k] = js(probs[obs[i], k], probs[obs[i - 1], k]).mean()
    return out


# ── offset change and persistence ─────────────────────────────────────────────


def offset_change(offset: np.ndarray, window: int) -> np.ndarray:
    """c_t = d_t minus the mean of the node's `window` previous contacts.

    The level of a node's offset is confounded by legitimate heterogeneity; a
    skewed node has a large offset that is *stable*. Its change against the
    node's own trailing mean is what a moving feed produces and a stable skew
    does not. Trailing means run over the node's own observed rounds, so a
    boat that surfaces every k rounds is compared with its own last `window`
    contacts.

    Args:
        offset: (rounds, clients, probes, classes) offsets from consensus.
        window: Number of previous contacts in the trailing mean.

    Returns:
        Array shaped like `offset`, NaN until `window` prior contacts exist.

    Raises:
        ValueError: If `window` is less than 1.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    K = offset.shape[1]
    out = np.full_like(offset, np.nan)
    for k in range(K):
        obs = _observed_rounds(offset, k)
        for i in range(window, len(obs)):
            trailing_mean = offset[obs[i - window : i], k].mean(axis=0)
            out[obs[i], k] = offset[obs[i], k] - trailing_mean
    return out


def persistence(vectors: np.ndarray) -> float:
    """‖mean_t v_t‖ / mean_t ‖v_t‖ over the leading axis, in [0, 1].

    Zero-mean noise across T rounds gives roughly 1/√T; a shift that points
    the same way every round gives 1.

    Args:
        vectors: (T, ...) vectors; trailing axes are flattened.

    Returns:
        The persistence, NaN if any vector is missing or all are zero.
    """
    v = np.asarray(vectors, dtype=np.float64).reshape(vectors.shape[0], -1)
    if np.isnan(v).any():
        return float("nan")
    norms = np.linalg.norm(v, axis=1)
    denom = norms.mean()
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(v.mean(axis=0)) / denom)


def persistence_series(change: np.ndarray, span: int) -> np.ndarray:
    """Persistence of the offset change over a node's trailing contacts.

    Each value covers the node's trailing `span` contacts ending at that
    contact.

    Args:
        change: (rounds, clients, ...) offset change, as from
            `offset_change`.
        span: Number of contacts per persistence window.

    Returns:
        (rounds, clients) persistence, NaN until `span` changes exist for
        that node.

    Raises:
        ValueError: If `span` is less than 2.
    """
    if span < 2:
        raise ValueError("span must be >= 2")
    R, K = change.shape[:2]
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(change, k)
        for i in range(span - 1, len(obs)):
            out[obs[i], k] = persistence(change[obs[i - span + 1 : i + 1], k])
    return out


# ── self-referenced level ─────────────────────────────────────────────────────


def self_referenced_level(level: np.ndarray, window: int) -> np.ndarray:
    """A node's level this round against its own trailing contacts.

    Computed as (level_t minus trailing mean) / trailing SD over the node's
    own previous `window` contacts.

    Added after the first stage-3 run, not before: a slow bias drift doubled
    the target node's level while its cross-node z *fell*, because the fleet
    threshold is set by the most-skewed healthy node and the drift moved the
    target toward the fleet median. The level is confounded across nodes but
    not across a node's own history. Thresholds for this statistic still come
    from healthy runs only.

    Args:
        level: (rounds, clients) level series.
        window: Number of previous contacts in the trailing statistics.

    Returns:
        (rounds, clients) self-referenced z, NaN until `window` prior
        contacts exist, and where the trailing SD is zero.

    Raises:
        ValueError: If `window` is less than 2.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    R, K = level.shape
    out = np.full((R, K), np.nan)
    for k in range(K):
        obs = _observed_rounds(level[:, :, None], k)
        for i in range(window, len(obs)):
            trailing = level[obs[i - window : i], k]
            sd = trailing.std(ddof=1)
            if sd > 0:
                out[obs[i], k] = (level[obs[i], k] - trailing.mean()) / sd
    return out


# ── cumulative sum against a fixed reference epoch ────────────────────────────


class _NodeReference(NamedTuple):
    """One node's reference-epoch footprint."""

    obs: np.ndarray  # every round the node reported
    ref: list  # the subset of those inside the reference epoch
    mu: float  # the node's own mean level over `ref`


def _reference_contacts(
    level: np.ndarray, ref_rounds: set[int]
) -> dict[int, _NodeReference]:
    """Node -> reference footprint, for nodes with two or more ref contacts.

    A node with fewer cannot have a spread estimated at all, so it is left out
    and `cusum` returns NaN for it throughout.
    """
    refs = {}
    for c in range(level.shape[1]):
        obs = _observed_rounds(level[:, :, None], c)
        ref = [r for r in obs if r in ref_rounds]
        if len(ref) >= 2:
            refs[c] = _NodeReference(obs, ref, level[ref, c].mean())
    return refs


def _pooled_reference_sd(
    level: np.ndarray,
    refs: dict[int, _NodeReference],
    pool: np.ndarray,
    min_ref: int,
) -> dict:
    """Pooled reference SD per platform class, only for classes that need one.

    A class whose every node has at least `min_ref` reference contacts is left
    out, so its nodes keep their own per-node SD. For any other class the SD is
    the root mean square of every member's deviations from its own reference
    mean, with one degree of freedom spent per node. A class with no member in
    `refs` maps to 0.0; no node ever looks that entry up.
    """
    pooled: dict = {}
    for cls in np.unique(pool):
        members = [c for c in refs if pool[c] == cls]
        if members and min(len(refs[c].ref) for c in members) >= min_ref:
            continue  # well-sampled class: per-node SD
        devs = np.concatenate(
            [level[refs[c].ref, c] - refs[c].mu for c in members]
            or [np.array([])]
        )
        dof = len(devs) - len(members)
        pooled[cls] = float(np.sqrt((devs**2).sum() / dof)) if dof > 0 else 0.0
    return pooled


def _cusum_path(
    level_col: np.ndarray,
    node: _NodeReference,
    sd: float,
    stop: int,
    k: float,
    two_sided: bool,
) -> Iterator[tuple[int, float]]:
    """Yield (round, CUSUM value) over the node's own contacts from `stop` on.

    A silent round is never visited, so it neither adds nor decays the sum.
    """
    s_up = s_dn = 0.0
    for r in node.obs:
        if r < stop:
            continue
        z = (level_col[r] - node.mu) / sd
        s_up = max(0.0, s_up + z - k)
        s_dn = max(0.0, s_dn - z - k)
        yield r, max(s_up, s_dn) if two_sided else s_up


def cusum(
    level: np.ndarray,
    reference: slice,
    k: float = 0.5,
    pool: np.ndarray | None = None,
    min_ref: int = 5,
    two_sided: bool = False,
) -> np.ndarray:
    """One-sided CUSUM (Page, 1954) of a node's level against a fixed epoch.

    Each node is measured against its OWN mean and SD over a fixed
    `reference` epoch:

        S_t = max(0, S_{t-1} + (level_t - μ_ref) / σ_ref - k)

    A step shows up in a sliding-window z; a slow ramp does not, because each
    round's increment sits inside the node's own noise. CUSUM accumulates the
    small consistent excess and is the standard estimator for "directional
    and persists across rounds". The reference epoch is the same fixed rounds
    for every node and every run, so it never sees an onset. Reference
    statistics and the accumulation run over the node's own contacts; a
    silent round neither adds nor decays.

    Args:
        level: (rounds, clients) level series.
        reference: Rounds forming the reference epoch.
        k: The allowance in SD units; 0.5 is the textbook default and is not
            tuned here.
        pool: Optional (clients,) array of platform-class labels. The
            reference MEAN is always per node (heterogeneity gives every
            node its own level), but the reference SD is pooled within a
            class: the deviations of every node in the class from its own
            mean, together. A boat that reports every third round has three
            reference contacts, and a spread estimated from three points is
            what blew one healthy boat's ceiling to 26 in the submarine arm.
            Pooling is the standard remedy, decided from healthy runs. It
            applies only where it is needed: a class whose every node has at
            least `min_ref` reference contacts keeps per-node SDs, so a dense
            fleet is unchanged by passing `pool`; a class with a starved node
            is pooled.
        min_ref: Reference contacts every node of a class needs for the
            class to keep per-node SDs.
        two_sided: Also accumulate deviations BELOW the reference and report
            the larger of the two sums. Added after the stage-4 grid: a gain
            fault moved the node toward the fleet, a decrease in its level,
            which a one-sided accumulator cannot see. Its healthy ceiling is
            judged on the healthy runs before it is scored on any fault.

    Returns:
        (rounds, clients) CUSUM values. NaN through the reference epoch, for
        nodes with fewer than two reference contacts, and wherever the node
        is absent.

    Raises:
        ValueError: If the reference epoch spans fewer than two rounds.
    """
    R, K = level.shape
    epoch = range(R)[reference]
    if len(epoch) < 2:
        raise ValueError("reference epoch must span >= 2 rounds")
    refs = _reference_contacts(level, set(epoch))
    pool = None if pool is None else np.asarray(pool)
    pooled_sd = (
        {} if pool is None else _pooled_reference_sd(level, refs, pool, min_ref)
    )

    out = np.full((R, K), np.nan)
    for c, node in refs.items():
        sd = None if pool is None else pooled_sd.get(pool[c])
        if sd is None:
            sd = level[node.ref, c].std(ddof=1)
        if sd <= 0:
            continue
        path = _cusum_path(level[:, c], node, sd, epoch.stop, k, two_sided)
        for r, s in path:
            out[r, c] = s
    return out


# ── instrument self-check ─────────────────────────────────────────────────────


def check_reports(
    reports: list[ProbeReport],
    battery: ProbeBattery,
    previous: dict[int, torch.Tensor] | None = None,
    row_tol: float = 1e-4,
) -> list[str]:
    """Integrity issues on the probe pathway itself.

    Independent of what the scores say. A degraded scorer and a degraded feed
    look alike in naive monitoring; this check is how they are told apart.

    Args:
        reports: One round's probe reports.
        battery: The battery that was served.
        previous: Maps client_id to that client's report from the prior
            round, for the replay check.
        row_tol: Largest tolerated deviation of a row sum from 1.

    Returns:
        Human-readable issues; an empty list means clean.
    """
    issues: list[str] = []
    shape = (battery.n_probes, battery.n_classes)
    for rep in reports:
        who = f"client {rep.client_id} round {rep.round_idx}"
        if rep.battery_fingerprint != battery.fingerprint:
            issues.append(
                f"{who}: battery fingerprint {rep.battery_fingerprint}"
                f" != served {battery.fingerprint}"
            )
        p = rep.probs
        if tuple(p.shape) != shape:
            issues.append(f"{who}: shape {tuple(p.shape)} != {shape}")
            continue
        if not torch.isfinite(p).all():
            issues.append(f"{who}: non-finite values")
            continue
        row_err = float((p.sum(dim=1) - 1.0).abs().max())
        if row_err > row_tol:
            issues.append(
                f"{who}: rows do not sum to 1 (max err {row_err:.2e})"
            )
        if (
            previous is not None
            and rep.client_id in previous
            and torch.equal(p, previous[rep.client_id])
        ):
            issues.append(
                f"{who}: report is bitwise identical to previous round"
                " (replay)"
            )
    return issues


def scorer_is_deterministic(
    model: torch.nn.Module, battery: ProbeBattery
) -> bool:
    """Scoring the same state twice must give bitwise-equal rows.

    Args:
        model: Model to score.
        battery: Probe battery to score it on.

    Returns:
        True if two scorings are bitwise equal.
    """
    first = score_probes(model, battery)
    second = score_probes(model, battery)
    return torch.equal(first, second)


# ── healthy characterisation ──────────────────────────────────────────────────


def _nan_stat(fn: Callable[[np.ndarray], Any], a: np.ndarray) -> float:
    """Apply `fn` over the non-NaN entries; NaN (silently) if there are none."""
    a = np.asarray(a, dtype=np.float64)
    return float(fn(a[~np.isnan(a)])) if (~np.isnan(a)).any() else float("nan")


def characterise(
    probs: np.ndarray,
    sizes: list[int],
    tail_rounds: int = 10,
    window: int = 5,
    span: int = 5,
) -> list[dict]:
    """The healthy distribution of every statistic, per battery size.

    `probs` comes from one run over the largest battery; each size is its
    prefix. Everything is summarised over the final `tail_rounds` rounds so the
    training transient is excluded.

    Args:
        probs: (rounds, clients, probes, classes) probabilities from one
            healthy run over the largest battery.
        sizes: Battery sizes (probe-count prefixes) to characterise.
        tail_rounds: Final rounds to summarise over.
        window: Trailing window for `offset_change`.
        span: Span for `persistence_series`.

    Returns:
        One dict per size with:

          level_median       typical mean-JS to consensus
          level_round_sd     SD across the tail rounds within a node, mean over
                             nodes: sampling + training noise, the temporal
                             floor
          level_client_sd    SD across nodes within a round, mean over rounds:
                             heterogeneity + sampling noise
          level_z_max        largest |within-round z| any node reached
          self_median        typical temporal self-divergence
          persist_median/max persistence of the offset change (span rounds)
    """
    R = probs.shape[0]
    tail = slice(max(0, R - tail_rounds), R)
    out = []
    for n in sizes:
        sub = probs[:, :, :n, :]
        lv = level_series(sub)
        sd = self_divergence(sub)
        ch = offset_change(lv["offset"], window)
        ps = persistence_series(ch, span)

        level_tail = lv["level"][tail]
        round_sd = np.nanstd(level_tail, axis=0)
        client_sd = np.nanstd(level_tail, axis=1)
        out.append(
            {
                "n_probes": n,
                "level_median": _nan_stat(np.median, level_tail),
                "level_round_sd": _nan_stat(np.mean, round_sd),
                "level_client_sd": _nan_stat(np.mean, client_sd),
                "level_z_max": _nan_stat(np.max, np.abs(lv["z"][tail])),
                "self_median": _nan_stat(np.median, sd[tail]),
                "persist_median": _nan_stat(np.median, ps[tail]),
                "persist_max": _nan_stat(np.max, ps[tail]),
            }
        )
    return out
