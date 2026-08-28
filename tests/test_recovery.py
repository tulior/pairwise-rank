"""End-to-end synthetic smoke test.

A slower test than the unit tests. Generates a small balanced
tournament with a deterministic ground-truth judge, fits the
five-level ordinal model, and checks that:

  1. the recovered theta ordering matches the truth ordering
  2. the strongest item has P(best) close to 1
  3. the model samples without pathological diagnostics

Uses a fixed seed and deliberately large separations so the result
is robust. Does not check magnitudes (the model's prior on cutpoints
differs from the synthetic judge's hard cutoffs).

This test exercises the legacy 5-level path explicitly. The default
3-level path is exercised by examples/three_view.py.
"""
from __future__ import annotations

import random

from pairwise_rank import (
    VERDICT_LEVELS_5,
    run_tournament,
    fit_ordinal,
    summarize,
)


GROUND_TRUTH = {
    "alpha": 1.5,
    "beta": 0.5,
    "gamma": -0.5,
    "delta": -1.5,
}

CP = [-1.5, -0.5, 0.5, 1.5]


def synthetic_judge(left_id: str, right_id: str) -> str:
    diff = GROUND_TRUTH[right_id] - GROUND_TRUTH[left_id]
    rng = random.Random(hash((left_id, right_id)))
    eta = diff + rng.gauss(0, 0.2)
    if eta <= CP[0]:
        return "LEFT_STRONG"
    elif eta <= CP[1]:
        return "LEFT"
    elif eta <= CP[2]:
        return "TIE"
    elif eta <= CP[3]:
        return "RIGHT"
    else:
        return "RIGHT_STRONG"


# 12. synthetic end-to-end model smoke/recovery (5-level ordinal path)
def test_synthetic_end_to_end_recovers_ordering():
    candidate_ids = list(GROUND_TRUTH.keys())
    observations = run_tournament(
        candidate_ids, synthetic_judge, repeats=6,
        verdict_levels=VERDICT_LEVELS_5,
    )

    result = fit_ordinal(observations, item_ids=candidate_ids, draws=1000, tune=1500, chains=4, seed=0)
    s = summarize(result)

    fit_order = sorted(s["per_item"], key=lambda r: -r["theta_mean"])
    fit_ranking = [r["id"] for r in fit_order]
    truth_ranking = sorted(GROUND_TRUTH.keys(), key=lambda k: -GROUND_TRUTH[k])

    # Full ordering recovered
    assert fit_ranking == truth_ranking, (
        f"recovered ranking {fit_ranking} != truth ranking {truth_ranking}"
    )

    # Strongest item has high P(best)
    strongest = truth_ranking[0]
    strongest_row = next(r for r in s["per_item"] if r["id"] == strongest)
    assert strongest_row["p_best"] > 0.9, (
        f"P(best) for strongest item {strongest} = {strongest_row['p_best']:.3f}, expected > 0.9"
    )

    # Weakest item has expected rank near n
    weakest = truth_ranking[-1]
    weakest_row = next(r for r in s["per_item"] if r["id"] == weakest)
    n = len(s["per_item"])
    assert weakest_row["expected_rank"] > n - 1.5, (
        f"E[rank] for weakest {weakest} = {weakest_row['expected_rank']:.2f}, expected near {n}"
    )
