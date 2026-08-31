"""Simulation audit for the adaptive comparison-design layer.

Compare three comparison-design strategies on Davidson-generated data:

  A. complete round robin
  B. degree-6 fixed sparse graph
  C. adaptive frontier design (run_adaptive_best_set)

Measure coverage, calls, returned k, wall time, and the number of
adaptive batches (method C only). The headline correctness metric
is COVERAGE: across repeated simulations, the true best should be
inside the returned 95% credible set approximately 95% of the time.

The simulation reuses the existing transitive Davidson DGP from
``experiments/model_falsification/scripts/dgp.py``. We do NOT
introduce a new DGP; the existing one is the canonical source of
synthetic Davidson observations for this codebase.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

# Make src/ importable
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "model_falsification", "scripts"))

# Reuse the existing DGP
from dgp import dgp_transitive  # noqa: E402

from pairwise_rank import Observation  # noqa: E402
from pairwise_rank.btd import fit_btd, summarize_btd  # noqa: E402
from pairwise_rank.design import (  # noqa: E402
    AdaptiveBestSetConfig,
    make_sparse_bootstrap,
    credible_best_set,
)


# ---------------------------------------------------------------------------
# Verdict strings from the synthetic DGP
# ---------------------------------------------------------------------------

VERDICT_FROM_CODE = {0: "LEFT", 1: "TIE", 2: "RIGHT"}


# ---------------------------------------------------------------------------
# Per-cell BTD ground truth (used only for DGP generation, not for fitting)
# ---------------------------------------------------------------------------

def make_synthetic_judge(tournament, n_obs_per_directed_pair: int = 1):
    """Build a deterministic judge_fn over ``(left_id, right_id)`` for the
    orchestrator. The judge returns the same verdict for the same
    (left, right) by looking up the synthetic data.

    The orchestrator calls ``judge_fn(o.left, o.right)`` for each
    oriented observation, so the judge is keyed by (left_id, right_id).
    """
    by_pair: dict[tuple[str, str], str] = {}
    # tournament.left, tournament.right, tournament.verdict are int-coded
    item_ids = tournament.item_ids
    for li, ri, vc in zip(tournament.left, tournament.right, tournament.verdict):
        key = (item_ids[li], item_ids[ri])
        # If multiple observations exist for the same (left, right),
        # the first one wins (deterministic). This is fine for
        # a synthetic judge: the simulation is on a fixed ground truth.
        if key not in by_pair:
            by_pair[key] = VERDICT_FROM_CODE[int(vc)]
    def judge(left: str, right: str) -> tuple[str, str]:
        return by_pair.get((left, right), "TIE"), ""
    return judge


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    """Per-seed, per-method result."""
    method: str           # "round_robin" | "sparse" | "adaptive"
    n_items: int
    seed: int
    n_obs: int            # total observations collected
    n_unordered_pairs: int
    wall_time_s: float
    true_best_idx: int
    credible_set: tuple[str, ...]
    k: int
    coverage: int         # 1 if true_best in credible_set, else 0
    top1_recovery: int    # 1 if credible_set[0] == true_best, else 0
    n_adaptive_batches: int | None = None
    stopped_reason: str | None = None
    true_best_id: str = ""
    credible_set_top_id: str = ""


def _observations_to_lists(observations: Iterable[Observation], item_ids: list[str]):
    obs = [o for o in observations if o.verdict]
    return obs


def run_round_robin(tournament, seed: int, draws: int = 200, tune: int = 200,
                    chains: int = 2) -> MethodResult:
    """Method A: complete round robin. Collect K=1 verdict per orientation
    for every unordered pair. Fit BTD. Return the credible set.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    item_ids = list(tournament.item_ids)
    observations: list[Observation] = []
    # tournament.left, tournament.right, tournament.verdict are the
    # directed-pair observations for K=1. For round-robin we use the
    # full directed-pair set.
    for li, ri, vc in zip(tournament.left, tournament.right, tournament.verdict):
        observations.append(Observation(
            a=item_ids[int(li)], b=item_ids[int(ri)],
            left=item_ids[int(li)], right=item_ids[int(ri)],
            repeat=1,
            verdict=VERDICT_FROM_CODE[int(vc)],
        ))
    fit = fit_btd(observations, item_ids=item_ids, seed=seed,
                  draws=draws, tune=tune, chains=chains, target_accept=0.9)
    summary = summarize_btd(fit, observations, position_neutral=True)
    p_best = {row["id"]: float(row["p_best"]) for row in summary["per_item"]}
    s = credible_best_set(item_ids, [p_best[i] for i in item_ids], confidence=0.95)
    true_best_idx = int(np.argmax(tournament.true_theta))
    true_best_id = item_ids[true_best_idx]
    elapsed = time.time() - t0
    return MethodResult(
        method="round_robin",
        n_items=tournament.n, seed=seed,
        n_obs=len(observations), n_unordered_pairs=tournament.n * (tournament.n - 1) // 2,
        wall_time_s=elapsed,
        true_best_idx=true_best_idx, true_best_id=true_best_id,
        credible_set=s, k=len(s),
        coverage=int(true_best_id in s),
        top1_recovery=int(len(s) >= 1 and s[0] == true_best_id),
        n_adaptive_batches=None,
    )


def run_sparse(tournament, seed: int, degree: int = 6, draws: int = 200,
               tune: int = 200, chains: int = 2) -> MethodResult:
    """Method B: fixed sparse graph at the given degree. No adaptivity."""
    t0 = time.time()
    item_ids = list(tournament.item_ids)
    pairs = make_sparse_bootstrap(item_ids, degree=degree, seed=seed)
    by_oriented: dict[tuple[str, str], str] = {}
    for li, ri, vc in zip(tournament.left, tournament.right, tournament.verdict):
        by_oriented[(item_ids[int(li)], item_ids[int(ri)])] = VERDICT_FROM_CODE[int(vc)]
    observations: list[Observation] = []
    for a, b in pairs:
        for left, right in ((a, b), (b, a)):
            verdict = by_oriented.get((left, right), "TIE")
            observations.append(Observation(
                a=a, b=b, left=left, right=right, repeat=1, verdict=verdict,
            ))
    fit = fit_btd(observations, item_ids=item_ids, seed=seed,
                  draws=draws, tune=tune, chains=chains, target_accept=0.9)
    summary = summarize_btd(fit, observations, position_neutral=True)
    p_best = {row["id"]: float(row["p_best"]) for row in summary["per_item"]}
    s = credible_best_set(item_ids, [p_best[i] for i in item_ids], confidence=0.95)
    true_best_idx = int(np.argmax(tournament.true_theta))
    true_best_id = item_ids[true_best_idx]
    elapsed = time.time() - t0
    return MethodResult(
        method="sparse",
        n_items=tournament.n, seed=seed,
        n_obs=len(observations), n_unordered_pairs=len(pairs),
        wall_time_s=elapsed,
        true_best_idx=true_best_idx, true_best_id=true_best_id,
        credible_set=s, k=len(s),
        coverage=int(true_best_id in s),
        top1_recovery=int(len(s) >= 1 and s[0] == true_best_id),
        n_adaptive_batches=None,
    )


def run_adaptive(
    tournament,
    seed: int,
    bootstrap_degree: int = 6,
    batch_size: int = 64,
    max_unordered_pairs: int | None = None,
    stability_batches: int = 2,
    confidence: float = 0.95,
) -> MethodResult:
    """Method C: adaptive frontier design via run_adaptive_best_set."""
    t0 = time.time()
    # Lazy import (the orchestrator imports pymc)
    from pairwise_rank.design import run_adaptive_best_set
    item_ids = list(tournament.item_ids)
    judge = make_synthetic_judge(tournament)
    cfg = AdaptiveBestSetConfig(
        confidence=confidence,
        bootstrap_degree=bootstrap_degree,
        batch_size=batch_size,
        stability_batches=stability_batches,
        max_unordered_pairs=max_unordered_pairs,
        max_per_item_per_batch=1,
        seed=seed,
    )
    result = run_adaptive_best_set(item_ids, judge, config=cfg, repeats=1)
    s = result.credible_best_set
    true_best_idx = int(np.argmax(tournament.true_theta))
    true_best_id = item_ids[true_best_idx]
    elapsed = time.time() - t0
    return MethodResult(
        method="adaptive",
        n_items=tournament.n, seed=seed,
        n_obs=0,  # not exposed by orchestrator; use unordered_pairs_used as proxy
        n_unordered_pairs=result.unordered_pairs_used,
        wall_time_s=elapsed,
        true_best_idx=true_best_idx, true_best_id=true_best_id,
        credible_set=s, k=result.k,
        coverage=int(true_best_id in s),
        top1_recovery=int(len(s) >= 1 and s[0] == true_best_id),
        n_adaptive_batches=result.batches,
        stopped_reason=result.stopped_reason,
        credible_set_top_id=s[0] if s else "",
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

@dataclass
class MethodSummary:
    method: str
    n_items: int
    n_seeds: int
    coverage_mean: float
    coverage_ci95_lo: float
    coverage_ci95_hi: float
    avg_calls: float
    avg_k: float
    avg_wall_time_s: float
    top1_recovery_mean: float
    n_adaptive_batches_mean: float | None = None
    stopped_reasons: dict | None = None


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_method(results: list[MethodResult]) -> MethodSummary:
    n_seeds = len(results)
    coverage = np.array([r.coverage for r in results], dtype=float)
    p_hat = float(coverage.mean()) if n_seeds > 0 else 0.0
    lo, hi = wilson_ci(p_hat, n_seeds)
    avg_calls = float(np.mean([r.n_unordered_pairs for r in results]))
    avg_k = float(np.mean([r.k for r in results]))
    avg_wall = float(np.mean([r.wall_time_s for r in results]))
    top1 = float(np.mean([r.top1_recovery for r in results]))
    n_adaptive = [r.n_adaptive_batches for r in results
                  if r.n_adaptive_batches is not None]
    n_adaptive_mean = float(np.mean(n_adaptive)) if n_adaptive else None
    reasons: dict = {}
    for r in results:
        if r.stopped_reason is not None:
            reasons[r.stopped_reason] = reasons.get(r.stopped_reason, 0) + 1
    return MethodSummary(
        method=results[0].method if results else "?",
        n_items=results[0].n_items if results else 0,
        n_seeds=n_seeds,
        coverage_mean=p_hat,
        coverage_ci95_lo=lo,
        coverage_ci95_hi=hi,
        avg_calls=avg_calls,
        avg_k=avg_k,
        avg_wall_time_s=avg_wall,
        top1_recovery_mean=top1,
        n_adaptive_batches_mean=n_adaptive_mean,
        stopped_reasons=reasons or None,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="32,100",
                        help="comma-separated list of N values to simulate")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--out", default="results/sim_results.json")
    parser.add_argument("--max-orders", type=int, default=None,
                        help="max number of unordered pairs for adaptive method")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stability-batches", type=int, default=2)
    parser.add_argument("--bootstrap-degree", type=int, default=6)
    args = parser.parse_args(argv)
    n_list = [int(s) for s in args.n_list.split(",")]
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_results: list[MethodResult] = []
    for n in n_list:
        # small n
        max_orders = args.max_orders
        if max_orders is None:
            # Default budget: 4x the sparse-bootstrap graph size.
            max_orders = max(2 * n, 4 * (n * args.bootstrap_degree // 2))
        for seed in range(args.n_seeds):
            print(f"[N={n} seed={seed}] generating DGP", flush=True)
            t = dgp_transitive(n=n, K=1, seed=seed,
                               theta=np.linspace(1.5, -1.5, n))
            try:
                r_rr = run_round_robin(t, seed=seed)
                print(f"  round_robin: k={r_rr.k} cov={r_rr.coverage} t={r_rr.wall_time_s:.1f}s", flush=True)
                all_results.append(r_rr)
            except Exception as e:
                print(f"  round_robin FAILED: {e}", flush=True)
            try:
                r_sp = run_sparse(t, seed=seed, degree=args.bootstrap_degree)
                print(f"  sparse     : k={r_sp.k} cov={r_sp.coverage} t={r_sp.wall_time_s:.1f}s", flush=True)
                all_results.append(r_sp)
            except Exception as e:
                print(f"  sparse FAILED: {e}", flush=True)
            try:
                r_ad = run_adaptive(t, seed=seed,
                                    bootstrap_degree=args.bootstrap_degree,
                                    batch_size=args.batch_size,
                                    max_unordered_pairs=max_orders,
                                    stability_batches=args.stability_batches)
                print(f"  adaptive   : k={r_ad.k} cov={r_ad.coverage} t={r_ad.wall_time_s:.1f}s batches={r_ad.n_adaptive_batches}", flush=True)
                all_results.append(r_ad)
            except Exception as e:
                print(f"  adaptive FAILED: {e}", flush=True)

    # Aggregate
    summaries: list[MethodSummary] = []
    by_method: dict[tuple[str, int], list[MethodResult]] = defaultdict(list)
    for r in all_results:
        by_method[(r.method, r.n_items)].append(r)
    for key, rs in sorted(by_method.items()):
        s = summarize_method(rs)
        summaries.append(s)
        print(f"\n  N={s.n_items} method={s.method}: "
              f"coverage={s.coverage_mean:.2f} "
              f"[{s.coverage_ci95_lo:.2f}, {s.coverage_ci95_hi:.2f}] "
              f"calls={s.avg_calls:.0f} k={s.avg_k:.1f} t={s.avg_wall_time_s:.1f}s", flush=True)

    # Persist
    out = {
        "n_list": n_list,
        "n_seeds": args.n_seeds,
        "bootstrap_degree": args.bootstrap_degree,
        "summaries": [asdict(s) for s in summaries],
        "raw_results": [asdict(r) for r in all_results],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
