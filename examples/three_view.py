"""Three-view report example.

Demonstrates the routine report pattern: direct W/L/T, BTD ranking, M0
ordinal ranking, all on the same observations. The three views are
cross-checked: if all three put the same item first, the winner is
robust to modeling choice.

This example uses the synthetic judge so it is fully reproducible and
self-contained. Replace the judge with your own callable for real
use.

Run with:
    python examples/three_view.py
"""
from __future__ import annotations

import random
from pathlib import Path

from pairwise_rank import (
    run_tournament,
    three_view_report,
    print_three_view,
)


# 4 candidates, clear ordering: alpha > beta > gamma > delta
CANDIDATES = ["alpha", "beta", "gamma", "delta"]
GROUND_TRUTH = {
    "alpha": 1.5,
    "beta": 0.5,
    "gamma": -0.5,
    "delta": -1.5,
}


def deterministic_judge(left: str, right: str) -> tuple[str, str]:
    """Synthetic 5-level judge based on ground-truth strengths.

    Returns (verdict, reasoning). The reasoning is a short fake
    audit-trail string so the test exercises the (verdict, reasoning)
    return path.
    """
    diff = GROUND_TRUTH[right] - GROUND_TRUTH[left]
    # Add small jitter to make verdicts probabilistic
    rng = random.Random(hash((left, right)) & 0xFFFFFFFF)
    observed = diff + rng.uniform(-0.5, 0.5)
    if observed > 1.5:
        verdict = "RIGHT_STRONG"
    elif observed > 0.5:
        verdict = "RIGHT"
    elif observed > -0.5:
        verdict = "TIE"
    elif observed > -1.5:
        verdict = "LEFT"
    else:
        verdict = "LEFT_STRONG"
    reasoning = f"ground_truth_diff={diff:.2f}, observed={observed:.2f}"
    return verdict, reasoning


def main() -> None:
    # K=3 reps × both orientations × 6 pairs = 36 obs
    obs = run_tournament(CANDIDATES, deterministic_judge, repeats=3)
    print(f"Collected {len(obs)} observations\n")

    report = three_view_report(
        obs, draws=2000, tune=2500, chains=4, target_accept=0.99, seed=0,
    )
    print_three_view(report, label="synthetic four-candidate tournament")

    print()
    if report["top1"]["all_three_agree"]:
        print(f"All three views agree on top-1: {report['top1']['direct']!r}")
    else:
        print(
            f"DISAGREEMENT: direct={report['top1']['direct']!r}, "
            f"btd={report['top1']['btd']!r}, m0={report['top1']['m0']!r}"
        )

    print(f"BTD vs M0 θ correlation: r = {report['theta_corr_btd_m0']:.4f}")
    print(f"BTD vs M0 P(best) correlation: r = {report['pbest_corr_btd_m0']:.4f}")


if __name__ == "__main__":
    main()
