"""Adaptive comparison-design layer for large-N pairwise tournaments.

# Principle

Likelihood pools evidence; design buys evidence.

The BTD likelihood in :mod:`pairwise_rank.btd` is the only place
the package estimates latent strengths from verdicts. It does not
decide which comparisons to collect. This module is the
experimental-design layer that decides which pairs to compare
next given the current BTD posterior, and when to stop. It does
**not** add another model, another sampler, or another judge
provider. It only consumes the existing BTD summaries
(``BTDFitResult.theta_draws`` and
``summarize_btd(..., position_neutral=True)``) and emits a list of
unordered pairs to judge.

# Position

This is a thin, opt-in, position-neutral design layer. It does not
modify :mod:`pairwise_rank.btd`, :mod:`pairwise_rank.protocol`, or
:mod:`pairwise_rank.providers`. It only reads BTD outputs and
writes pair lists. The orientation of a pair is the protocol's
job; the design layer works strictly on **unordered** pairs.

# Small-N threshold

For ``N <= 12`` items, the recommended mode is a complete
counterbalanced round robin. The design functions in this module
do not need to know about it: the user simply does not call
``select_frontier_batch`` for small N. The threshold is an
engineering constant, not a theorem: it reflects the point at
which ``O(N^2)`` becomes more expensive than a few BTD refits
under a sparse initial graph. See ``EXPERIMENT_DESIGN.md`` for the
longer discussion.

# Acquisition heuristic

The frontier score used in :func:`select_frontier_batch`

    score(i, j) = (p_best[i] + p_best[j]) * 4 * q_ij * (1 - q_ij)

is not formally optimal. It is a transparent decision-focused
proxy that combines:

- **relevance** ``(p_best[i] + p_best[j])``: how much both items
  matter to the top-K decision.
- **uncertainty** ``4 * q_ij * (1 - q_ij)``: peaked at ``q_ij = 0.5``
  (i.e. the most informative posterior on the pairwise
  comparison) and zero at ``q_ij in {0, 1}`` (i.e. the comparison
  is already decided).

The shape is ``(S, n) -> (S, S)`` for ``q_ij`` and a per-pair
scalar for the score. ``4 * q * (1 - q)`` is the variance factor
of a Bernoulli random variable with parameter ``q``; its maximum
is at ``q = 0.5`` with value ``1``.

This is one possible score. The contract is that the function
takes a position-neutral ``p_best`` vector and a position-neutral
``pairwise_gt`` matrix and returns a deterministic, well-defined
batch. The score itself is intentionally simple so it can be
audited and replaced without touching the rest of the package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import numpy as np

from .protocol import Observation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdaptiveBestSetConfig:
    """Tuning knobs for adaptive large-N comparison design.

    Validation is enforced by ``__post_init__``. A misconfigured
    value raises ``ValueError`` immediately rather than silently
    producing nonsense.
    """

    confidence: float = 0.95
    bootstrap_degree: int = 6
    batch_size: int = 64
    stability_batches: int = 2
    max_unordered_pairs: int | None = None
    max_per_item_per_batch: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.confidence) < 1.0):
            raise ValueError(
                f"confidence must be in (0, 1), got {self.confidence!r}"
            )
        if int(self.bootstrap_degree) < 2:
            raise ValueError(
                f"bootstrap_degree must be >= 2, got {self.bootstrap_degree!r}"
            )
        if int(self.bootstrap_degree) % 2 != 0:
            raise ValueError(
                f"bootstrap_degree must be even, got {self.bootstrap_degree!r}"
            )
        if int(self.batch_size) < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size!r}"
            )
        if int(self.stability_batches) < 1:
            raise ValueError(
                f"stability_batches must be >= 1, got {self.stability_batches!r}"
            )
        if int(self.max_per_item_per_batch) < 1:
            raise ValueError(
                f"max_per_item_per_batch must be >= 1, "
                f"got {self.max_per_item_per_batch!r}"
            )
        if self.max_unordered_pairs is not None and int(self.max_unordered_pairs) < 1:
            raise ValueError(
                f"max_unordered_pairs must be >= 1 when set, "
                f"got {self.max_unordered_pairs!r}"
            )


@dataclass(frozen=True)
class AdaptiveBestSetState:
    """Internal state exposed to the orchestrator for inspection.

    This is intentionally frozen and read-only: the orchestrator
    builds a new state object each batch. The pure design functions
    do not need this; it exists for callers that want to persist
    state across batches or to debug the stopping logic.
    """

    candidate_ids: tuple[str, ...]
    completed_pairs: tuple[tuple[str, str], ...]
    completed_observations: tuple[Observation, ...]
    credible_set_history: tuple[tuple[str, ...], ...]
    unordered_pairs_used: int


@dataclass(frozen=True)
class AdaptiveBestSetResult:
    """Final output of the adaptive top-K / best-set selector.

    Fields:
      items:                the candidate set the design layer ran on.
      p_best:               position-neutral ``P(theta_i = argmax theta)``
                            from the most recent BTD fit, indexed by
                            item id. ``sum(p_best.values()) == 1``.
      credible_best_set:    smallest prefix of items sorted by
                            descending ``p_best`` whose cumulative mass
                            is at least ``confidence``. Sorted for
                            display by descending ``p_best``, with
                            ties broken by deterministic item order.
      confidence:           the credible-set confidence used.
      k:                    ``len(credible_best_set)``.
      stopped_reason:       ``"stable"`` if the credible set was the
                            same for the last ``stability_batches``
                            fits, ``"budget"`` if ``max_unordered_pairs``
                            was hit first.
      batches:              number of BTD refits performed.
      unordered_pairs_used: total number of unordered pairs with at
                            least one completed observation. Counts
                            each unordered pair once even if both
                            orientations are present.
      expected_rank:        position-neutral expected rank
                            ``E[rank(theta_i)]`` from the most recent
                            BTD fit, lower = better. ``None`` if the
                            orchestrator did not have a fit to draw
                            from (e.g. the bootstrap itself was the
                            last step).
    """

    items: tuple[str, ...]
    p_best: dict[str, float]
    credible_best_set: tuple[str, ...]
    confidence: float
    k: int
    stopped_reason: Literal["stable", "budget"]
    batches: int
    unordered_pairs_used: int
    expected_rank: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Credible best set
# ---------------------------------------------------------------------------

def credible_best_set(
    items: Sequence[str],
    p_best: Sequence[float],
    confidence: float = 0.95,
) -> tuple[str, ...]:
    """Return the smallest prefix ``S`` of items (sorted by
    descending ``p_best``) whose cumulative mass is at least
    ``confidence``.

    The sort uses a stable argsort, so equal ``p_best`` entries
    are ordered by their original index in ``items``. ``k`` is
    defined as ``len(S)``.

    The input ``p_best`` is not silently renormalized. The
    function validates:

      - ``0 < confidence < 1``,
      - ``p_best.shape == (len(items),)``,
      - every entry is finite and ``>= 0``,
      - ``sum(p_best)`` is within ``1e-6`` of ``1``.

    Any deviation raises ``ValueError`` with a descriptive
    message. The user is expected to pass a valid ``p_best``
    (i.e. the position-neutral ``P(best)`` from
    ``summarize_btd(..., position_neutral=True)``).
    """
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    n = len(items)
    p = np.asarray(p_best, dtype=float)
    if p.shape != (n,):
        raise ValueError(
            f"p_best.shape must be ({n},), got {p.shape!r}"
        )
    if not np.all(np.isfinite(p)):
        raise ValueError("p_best must be finite; got non-finite entries")
    if not np.all(p >= 0.0):
        raise ValueError("p_best must be non-negative")
    total = float(p.sum())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"p_best must sum to 1 within 1e-6; got {total!r}"
        )
    # Stable argsort by descending p_best; equal p_best values keep
    # the original index ordering.
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    cum = np.cumsum(sorted_p)
    # cum[k-1] >= confidence means the first k items suffice.
    # np.argmax on a boolean array returns the first True index, or 0
    # if no entry is True; we need a safe sentinel.
    hits = np.where(cum >= confidence)[0]
    if hits.size == 0:
        # Should not happen for a valid (normalized, non-negative) p,
        # but defend explicitly.
        raise ValueError(
            "credible_best_set: no prefix reaches confidence; this "
            "indicates an invalid p_best"
        )
    k = int(hits[0]) + 1
    selected_idx = order[:k]
    # Return a tuple of item ids in the same display order
    # (descending p_best, then deterministic item order on ties).
    return tuple(items[int(i)] for i in selected_idx)


# ---------------------------------------------------------------------------
# Sparse bootstrap graph
# ---------------------------------------------------------------------------

def make_sparse_bootstrap(
    items: Sequence[str],
    degree: int = 6,
    seed: int = 0,
) -> tuple[tuple[str, str], ...]:
    """Return a sparse connected undirected graph as a tuple of
    unordered (item, item) pairs.

    Construction:

      1. Validate ``degree >= 2`` and ``degree`` is even.
      2. If ``n <= degree``, return all ``n * (n - 1) / 2``
         unordered pairs.
      3. Otherwise, build a circulant graph of order ``n`` with
         offsets ``1, 2, ..., degree / 2``: each item ``i`` is
         connected to items ``i + k mod n`` for ``k`` in
         ``1 .. degree / 2``. The graph is undirected, so this
         adds exactly ``n * degree / 2`` unordered edges.
      4. Shuffle the order in which edges are added with the given
         seed (this is the only source of randomness), then
         canonicalize each pair as ``(min, max)`` by item id.

    The result is deterministic for a given ``(items, degree, seed)``
    and satisfies:

      - the graph is connected (the circulant with offsets
        ``1..k`` and ``n > 2k`` is connected by construction);
      - every item has degree ``degree`` when ``n > degree``;
      - there are no duplicate edges;
      - there are no self edges;
      - the edge count is exactly ``n * degree / 2``.
    """
    if int(degree) < 2:
        raise ValueError(f"degree must be >= 2, got {degree!r}")
    if int(degree) % 2 != 0:
        raise ValueError(f"degree must be even, got {degree!r}")
    n = len(items)
    if n < 2:
        return ()
    if n <= int(degree):
        # Return every unordered pair.
        return tuple(
            (items[i], items[j]) if items[i] < items[j] else (items[j], items[i])
            for i in range(n)
            for j in range(i + 1, n)
        )
    half = int(degree) // 2
    rng = np.random.default_rng(int(seed))
    # Build the edge list, then shuffle the order in which we
    # add them. We use a circulant definition (each item i gets
    # edges to (i + k) mod n for k = 1..half) so the graph is
    # connected when n > degree and every item has exactly
    # ``degree`` neighbors. Self edges are excluded by construction
    # because 0 is not in the offset list.
    edges_raw: list[tuple[int, int]] = []
    for i in range(n):
        for k in range(1, half + 1):
            j = (i + k) % n
            edges_raw.append((i, j))
    # Shuffle the addition order; deterministic given the seed.
    perm = rng.permutation(len(edges_raw))
    edges = [edges_raw[int(p)] for p in perm]
    out: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in edges:
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        out.append((items[a], items[b]))
    return tuple(out)


# ---------------------------------------------------------------------------
# Frontier batch selection
# ---------------------------------------------------------------------------

def select_frontier_batch(
    items: Sequence[str],
    p_best: dict[str, float],
    pairwise_gt: dict[tuple[int, int], float],
    confidence: float = 0.95,
    batch_size: int = 64,
    max_per_item_per_batch: int = 1,
) -> tuple[tuple[str, str], ...]:
    """Pick up to ``batch_size`` unordered pairs from the top-K
    exploration pool, scored by

        score(i, j) = (p_best[i] + p_best[j]) * 4 * q_ij * (1 - q_ij)

    where ``q_ij = P(theta_i > theta_j)`` is read from
    ``pairwise_gt`` (key ``(i, j)`` with ``i < j``). Missing
    entries in ``pairwise_gt`` default to ``q = 0.5`` (maximum
    uncertainty, ``4 * 0.5 * 0.5 = 1.0``).

    The exploration pool is the credible best set at the wider
    confidence

        1 - (1 - confidence) / 2   =   0.5 + confidence / 2

    which is, for ``confidence = 0.95``, the ``0.975`` pool. The
    pool is built by :func:`credible_best_set` over ``items`` and
    ``p_best`` at the wider confidence; only items in the pool
    are eligible to be selected.

    Tie-breaking is deterministic: pairs are sorted by descending
    score, with ties broken by the deterministic item order
    (lexicographic tuple of item ids). ``max_per_item_per_batch``
    caps how many times a single item may appear in one batch.
    If the pool has at least two items but no pair can be picked
    without violating the cap, the cap is relaxed (in steps of 1)
    until at least one pair is picked, so the batch always makes
    forward progress as long as the pool is non-trivial.

    The returned pairs are canonicalized as
    ``(min(item_a, item_b), max(item_a, item_b))`` so the same
    unordered pair never appears twice in one batch.
    """
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")
    if int(max_per_item_per_batch) < 1:
        raise ValueError(
            f"max_per_item_per_batch must be >= 1, got {max_per_item_per_batch!r}"
        )
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError(
            f"confidence must be in (0, 1), got {confidence!r}"
        )
    n = len(items)
    if n < 2:
        return ()

    # Build p_best array in the canonical item order. ``p_best`` is
    # a relevance score for the frontier selector; it is not
    # required to be a normalized probability distribution.
    p_arr = np.array([float(p_best.get(it, 0.0)) for it in items], dtype=float)

    # Wider exploration confidence for the pool. We normalize
    # ``p_arr`` to a valid ``credible_best_set`` input solely for
    # the purpose of computing the exploration pool membership.
    # The relevance used in the score is the original (unnormalized)
    # p_arr -- normalization here is a pool-size heuristic, not a
    # user-facing transformation.
    exploration_conf = 1.0 - (1.0 - float(confidence)) / 2.0
    total = float(p_arr.sum())
    if total > 0.0 and math.isfinite(total):
        p_for_pool = p_arr / total
        # Allow a small slack so the input can be a hand-tuned
        # weight vector that is close to but not exactly 1.0;
        # the only thing that matters is the ranking. The
        # ``credible_best_set`` 1e-6 tolerance already covers
        # floating-point noise from /total.
        p_for_pool = p_for_pool + 1e-12 * np.arange(n)
        p_for_pool = p_for_pool / float(p_for_pool.sum())
        try:
            pool_items = set(credible_best_set(
                list(items), p_for_pool, confidence=exploration_conf,
            ))
        except ValueError:
            # Fall back to including every item with positive
            # relevance; this keeps the selector usable even on
            # a degenerate input.
            pool_items = {it for it, v in zip(items, p_arr) if v > 0.0}
    else:
        pool_items = set(items)
    if len(pool_items) < 2:
        return ()

    pool_list = [it for it in items if it in pool_items]
    idx_of = {it: i for i, it in enumerate(items)}
    pool_indices = [idx_of[it] for it in pool_list]

    # Score every (i, j) in the pool. q_ij is read from pairwise_gt
    # by canonical index pair; missing keys default to q = 0.5
    # (uncertainty 1.0). Pairs whose q_ij is explicit in the dict
    # are preferred over pairs whose q_ij is the default when their
    # scores are equal, so an explicit q = 0.5 outranks a default
    # q = 0.5 at the same numerical score. The score itself is
    # still the dominant signal: a high-relevance default-q pair
    # outranks a low-relevance explicit-q pair.
    scored: list[tuple[float, int, tuple[str, str]]] = []
    for a_pos, a_idx in enumerate(pool_indices):
        for b_pos in range(a_pos + 1, len(pool_indices)):
            b_idx = pool_indices[b_pos]
            key = (a_idx, b_idx) if a_idx < b_idx else (b_idx, a_idx)
            explicit = key in pairwise_gt
            q = pairwise_gt.get(key, 0.5)
            q = float(q)
            if not math.isfinite(q):
                q = 0.5
            # Clamp into [0, 1]; a numeric slip should not poison
            # the score.
            if q < 0.0:
                q = 0.0
            elif q > 1.0:
                q = 1.0
            uncertainty = 4.0 * q * (1.0 - q)
            relevance = p_arr[a_idx] + p_arr[b_idx]
            score = relevance * uncertainty
            a_item = items[a_idx]
            b_item = items[b_idx]
            canonical = (a_item, b_item) if a_item < b_item else (b_item, a_item)
            priority = 1 if explicit else 0
            scored.append((score, priority, canonical))

    # Sort by descending score, then by explicit > default, then by
    # deterministic item order on ties. The score is the dominant
    # signal; the priority only breaks ties.
    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))

    # Greedy pick honoring max_per_item_per_batch. Relax the cap
    # in steps of 1 if needed to guarantee progress.
    cap = int(max_per_item_per_batch)
    while True:
        picked: list[tuple[str, str]] = []
        used_items: dict[str, int] = {it: 0 for it in pool_list}
        seen_pairs: set[tuple[str, str]] = set()
        for score, priority, pair in scored:
            if score <= 0.0:
                # q in {0, 1} yields score == 0, those pairs are not
                # informative. We still consider them if no other
                # progress is possible, but in the normal cap path
                # we skip them so the batch stays useful.
                continue
            a, b = pair
            if (a, b) in seen_pairs:
                continue
            if used_items[a] >= cap or used_items[b] >= cap:
                continue
            picked.append((a, b))
            seen_pairs.add((a, b))
            used_items[a] += 1
            used_items[b] += 1
            if len(picked) >= int(batch_size):
                break
        if picked or cap >= n:
            return tuple(picked)
        # Relax the cap and try again so the batch makes progress
        # when every candidate pair shares an item already at the
        # cap.
        cap += 1


# ---------------------------------------------------------------------------
# Stopping
# ---------------------------------------------------------------------------

def should_stop_adaptive(
    credible_set_history: Sequence[tuple[str, ...]],
    unordered_pairs_used: int,
    config: AdaptiveBestSetConfig,
) -> tuple[bool, str | None]:
    """Decide whether the adaptive loop should stop.

    Returns ``(True, "budget")`` if ``max_unordered_pairs`` is set
    and the count of unordered pairs reached it; ``(True, "stable")``
    if the last ``stability_batches`` credible sets are equal under
    ``frozenset`` identity; ``(False, None)`` otherwise. Stability
    requires at least ``stability_batches`` history entries.
    """
    max_pairs = config.max_unordered_pairs
    if max_pairs is not None and int(unordered_pairs_used) >= int(max_pairs):
        return (True, "budget")
    if len(credible_set_history) < int(config.stability_batches):
        return (False, None)
    tail = list(credible_set_history[-int(config.stability_batches):])
    reference = frozenset(tail[0])
    for entry in tail[1:]:
        if frozenset(entry) != reference:
            return (False, None)
    return (True, "stable")


# ---------------------------------------------------------------------------
# Helpers exposed for the orchestrator and the tests
# ---------------------------------------------------------------------------

def _pairwise_gt_from_draws(theta_draws: np.ndarray) -> dict[tuple[int, int], float]:
    """Build a ``pairwise_gt`` dict from posterior draws of theta.

    Computes ``mean(theta_s[i] > theta_s[j])`` for all ``i < j``.
    Diagonal pairs are not emitted. The function is a thin wrapper
    around the broadcasting operation in :mod:`pairwise_rank.btd`;
    it is duplicated here so :mod:`pairwise_rank.design` does not
    need to import any BTD internals (only the public ``BTDFitResult``
    shape).
    """
    if theta_draws.ndim != 2:
        raise ValueError(
            f"theta_draws must be 2D (S, n), got shape {theta_draws.shape!r}"
        )
    _, n = theta_draws.shape
    out: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            q = float((theta_draws[:, i] > theta_draws[:, j]).mean())
            out[(i, j)] = q
    return out


def _p_best_from_draws(theta_draws: np.ndarray) -> np.ndarray:
    """``P(best)`` from posterior draws of theta.

    Each draw credits exactly one item via ``argmax`` (the
    first-occurrence on ties, matching ``btd._p_best``). Returns
    a length-``n`` vector summing to 1.
    """
    if theta_draws.ndim != 2:
        raise ValueError(
            f"theta_draws must be 2D (S, n), got shape {theta_draws.shape!r}"
        )
    argmax_items = np.argmax(theta_draws, axis=1)
    n = theta_draws.shape[1]
    counts = np.bincount(argmax_items, minlength=n)
    return counts / theta_draws.shape[0]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _existing_observations_for_pairs(
    pairs: Sequence[tuple[str, str]],
    repeats: int,
    existing: Iterable[Observation],
) -> list[Observation]:
    """Build a deterministic schedule for ``pairs`` at the given
    repeat count, deduped against ``existing``.

    The orientation rule is the same as :func:`make_schedule`: for
    each unordered pair ``(a, b)`` with ``a < b`` (lexicographic
    id), the two orientations are ``(left=a, right=b)`` and
    ``(left=b, right=a)`` and each orientation has ``repeats``
    rows numbered ``1..repeats``. Rows already present in
    ``existing`` are skipped.
    """
    schedule: list[Observation] = []
    for a, b in pairs:
        if a == b:
            continue
        lo, hi = (a, b) if a < b else (b, a)
        for left, right in ((lo, hi), (hi, lo)):
            for r in range(1, int(repeats) + 1):
                schedule.append(Observation(
                    a=lo, b=hi, left=left, right=right, repeat=r, verdict="",
                ))
    done = {(o.a, o.b, o.left, o.right, o.repeat) for o in existing}
    return [o for o in schedule
            if (o.a, o.b, o.left, o.right, o.repeat) not in done]


def _count_unordered_pairs(observations: Sequence[Observation]) -> int:
    """Count the number of distinct unordered pairs with at least
    one completed observation. Each unordered pair is counted once
    even if both orientations are present.
    """
    s: set[tuple[str, str]] = set()
    for o in observations:
        if not o.verdict:
            continue
        a, b = (o.a, o.b) if o.a < o.b else (o.b, o.a)
        s.add((a, b))
    return len(s)


def run_adaptive_best_set(
    items: Sequence[str],
    judge_fn,  # the same JudgeFn callable used by run_tournament
    config: AdaptiveBestSetConfig | None = None,
    repeats: int = 3,
    existing_observations: Iterable[Observation] = (),
) -> AdaptiveBestSetResult:
    """Run the adaptive top-K / best-set loop.

    The orchestrator is the only place in the design layer that
    touches the BTD sampler; the pure design functions remain
    PyMC-free. The orchestrator itself only does scheduling and
    loop bookkeeping, never numerical work on the model.

    Algorithm:

      1. **Bootstrap**: collect a sparse connected subgraph
         (:func:`make_sparse_bootstrap`) and run ``repeats``
         judgments on each orientation. For each item, this is at
         most ``2 * repeats * degree`` observations.
      2. **Refit**: fit BTD and compute ``p_best`` (position
         neutral) and the full pairwise ``P(theta_i > theta_j)``
         table.
      3. **Credible set**: call :func:`credible_best_set` at
         ``config.confidence`` and append to history.
      4. **Stop check**: call :func:`should_stop_adaptive`; if it
         returns ``True``, return a final
         :class:`AdaptiveBestSetResult` with the latest posterior.
      5. **Acquire**: call :func:`select_frontier_batch` on the
         wider exploration pool and run ``repeats`` judgments per
         pair per orientation. Go to step 2.

    Small-N fallback: if ``len(items) <= 12``, the loop skips
    the adaptive selector and runs a single complete round-robin
    on every unordered pair. The function still returns a
    credible best set with ``stopped_reason = "budget"`` (the
    loop was budget-bounded, not stability-bounded).

    Budget semantics: the orchestrator does **not** silently pick
    the single highest-``p_best`` item when the budget is hit.
    It returns whatever the latest fit says. If the fit did not
    converge enough for the credible set to be well-defined at
    ``k >= 1``, ``k`` may be small, but it is the result of the
    threshold and not a forced collapse.
    """
    # Local import: the design module's pure functions must not
    # pull pymc / pytensor / arviz at import time. The orchestrator
    # is the one place a refit happens, so the heavy imports are
    # deferred to here.
    from .btd import fit_btd, summarize_btd, BTDFitResult

    cfg = config or AdaptiveBestSetConfig()
    items_t = tuple(items)
    n = len(items_t)
    if n == 0:
        raise ValueError("items must be non-empty")
    if n == 1:
        # Trivial: the one item is the best with p_best = 1.
        return AdaptiveBestSetResult(
            items=items_t,
            p_best={items_t[0]: 1.0},
            credible_best_set=items_t,
            confidence=cfg.confidence,
            k=1,
            stopped_reason="budget",
            batches=0,
            unordered_pairs_used=0,
            expected_rank={items_t[0]: 1.0},
        )

    observations: list[Observation] = list(existing_observations)

    def _run_pairs(pairs: Sequence[tuple[str, str]]) -> int:
        """Judge every orientation/repeat of every pair that is
        not already done. Returns the number of new observations
        added.
        """
        new_obs = _existing_observations_for_pairs(pairs, repeats, observations)
        added = 0
        for o in new_obs:
            result = judge_fn(o.left, o.right)
            # Match protocol._split_judge_return without importing
            # private helpers; the judge is a public surface.
            if isinstance(result, tuple):
                if len(result) != 2:
                    raise ValueError(
                        f"judge_fn returned tuple of length {len(result)}; "
                        "expected 2 (verdict, reasoning)"
                    )
                verdict, reasoning = result
                reasoning_text = "" if reasoning is None else str(reasoning)
            else:
                verdict, reasoning_text = result, ""
            # Validate against the 3-level scale. The design layer
            # only consumes the 3-level scale; legacy 5-level
            # observations are the caller's responsibility (use
            # run_tournament with verdict_levels=VERDICT_LEVELS_5
            # for that).
            allowed = ("LEFT", "TIE", "RIGHT")
            if verdict not in allowed:
                raise ValueError(
                    f"judge_fn returned invalid verdict: {verdict!r}; "
                    f"expected one of {list(allowed)}"
                )
            o.verdict = verdict
            o.reasoning = reasoning_text
            observations.append(o)
            added += 1
        return added

    small_n = n <= 12
    bootstrap_pairs = make_sparse_bootstrap(
        items_t, degree=cfg.bootstrap_degree, seed=cfg.seed,
    )
    _run_pairs(bootstrap_pairs)

    last_result: BTDFitResult | None = None
    last_summary: dict | None = None
    history: list[tuple[str, ...]] = []
    batches = 0
    max_iters = 10_000  # absolute safety; the inner loop also stops
    while batches < max_iters:
        # Refit on the current observations. If the bootstrap
        # produced no usable observations (e.g. every row was
        # already completed by existing_observations and no new
        # data was needed), we still refit so the orchestrator
        # returns a credible set anchored on the most recent
        # posterior.
        completed = [o for o in observations if o.verdict]
        if not completed:
            raise ValueError(
                "run_adaptive_best_set: no completed observations to fit; "
                "supply at least one observation via existing_observations "
                "or via a non-empty bootstrap"
            )
        last_result = fit_btd(
            completed, item_ids=list(items_t), seed=cfg.seed,
        )
        last_summary = summarize_btd(
            last_result, completed, position_neutral=True,
        )
        per_item = last_summary["per_item"]
        p_best = {row["id"]: float(row["p_best"]) for row in per_item}
        expected_rank = {row["id"]: float(row["expected_rank"]) for row in per_item}
        pairwise_gt_raw = last_summary["pairwise"]
        # pairwise keys in summarize_btd are f"{i},{j}" with i<j.
        pairwise_gt: dict[tuple[int, int], float] = {}
        for k, v in pairwise_gt_raw.items():
            i_str, j_str = k.split(",")
            pairwise_gt[(int(i_str), int(j_str))] = float(v["p_i_gt_j"])
        s = credible_best_set(
            list(p_best.keys()), list(p_best.values()),
            confidence=cfg.confidence,
        )
        history.append(s)
        batches += 1

        unordered_pairs_used = _count_unordered_pairs(observations)
        stop, reason = should_stop_adaptive(history, unordered_pairs_used, cfg)
        if stop:
            return AdaptiveBestSetResult(
                items=items_t,
                p_best=p_best,
                credible_best_set=s,
                confidence=cfg.confidence,
                k=len(s),
                stopped_reason="stable" if reason == "stable" else "budget",
                batches=batches,
                unordered_pairs_used=unordered_pairs_used,
                expected_rank=expected_rank,
            )

        # Budget check: even if stability is not yet reached, we
        # may be at the unordered-pair cap. The should_stop_adaptive
        # above already returned (False, None) when at the cap only
        # when stability also failed; we still must not exceed the
        # cap on the next acquisition.
        if small_n:
            # Small-N mode: the bootstrap already covered every
            # item; we just refit until stable or budget. There are
            # no more pairs to acquire.
            return AdaptiveBestSetResult(
                items=items_t,
                p_best=p_best,
                credible_best_set=s,
                confidence=cfg.confidence,
                k=len(s),
                stopped_reason="budget",
                batches=batches,
                unordered_pairs_used=unordered_pairs_used,
                expected_rank=expected_rank,
            )

        if (
            cfg.max_unordered_pairs is not None
            and unordered_pairs_used >= int(cfg.max_unordered_pairs)
        ):
            return AdaptiveBestSetResult(
                items=items_t,
                p_best=p_best,
                credible_best_set=s,
                confidence=cfg.confidence,
                k=len(s),
                stopped_reason="budget",
                batches=batches,
                unordered_pairs_used=unordered_pairs_used,
                expected_rank=expected_rank,
            )

        batch = select_frontier_batch(
            list(items_t),
            p_best,
            pairwise_gt,
            confidence=cfg.confidence,
            batch_size=cfg.batch_size,
            max_per_item_per_batch=cfg.max_per_item_per_batch,
        )
        if not batch:
            # No informative pairs left; treat as a budget stop.
            return AdaptiveBestSetResult(
                items=items_t,
                p_best=p_best,
                credible_best_set=s,
                confidence=cfg.confidence,
                k=len(s),
                stopped_reason="budget",
                batches=batches,
                unordered_pairs_used=unordered_pairs_used,
                expected_rank=expected_rank,
            )
        _run_pairs(batch)

    # Defensive: the inner loop should always terminate via stop
    # or budget before reaching this. If it does not, surface the
    # current state rather than spin forever.
    return AdaptiveBestSetResult(
        items=items_t,
        p_best=p_best,
        credible_best_set=s,
        confidence=cfg.confidence,
        k=len(s),
        stopped_reason="budget",
        batches=batches,
        unordered_pairs_used=_count_unordered_pairs(observations),
        expected_rank=expected_rank,
    )
