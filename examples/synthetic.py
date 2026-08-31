"""Synthetic example: full pipeline with a deterministic ground-truth judge.

Run with:
    python examples/synthetic.py

Generates a small tournament over 4 fake candidates using a
deterministic 3-level judge, writes observations to JSONL, fits the
default BTD model, and prints a summary. The judge here is a
stand-in: replace it with your own callable for real use.

This example uses the default 3-level verdict scale (LEFT, TIE, RIGHT).
Legacy 5-level observations on disk load without migration;
`fit_btd` and `direct_summary` collapse STRONG into ordinary wins
and losses internally. No 5-level inference is performed.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from pairwise_rank import (
    run_tournament,
    save_observations_jsonl,
    load_observations_jsonl,
    fit_btd,
    summarize_btd,
    direct_summary,
)


GROUND_TRUTH = {
    "alpha": 1.5,
    "beta": 0.5,
    "gamma": -0.5,
    "delta": -1.5,
}

# Cutpoints for the synthetic judge: -0.5 (LEFT vs TIE), +0.5 (TIE vs RIGHT)
CP = [-0.5, 0.5]


def synthetic_judge(left_id: str, right_id: str) -> str:
    """Deterministic stand-in 3-level judge. The strength difference
    drives eta; small noise is added so the synthetic data has some
    disagreement across repeats. Replace with your own callable for
    real use.
    """
    diff = GROUND_TRUTH[right_id] - GROUND_TRUTH[left_id]
    rng = random.Random(hash((left_id, right_id)))
    eta = diff + rng.gauss(0, 0.2)
    if eta <= CP[0]:
        return "LEFT"
    elif eta <= CP[1]:
        return "TIE"
    else:
        return "RIGHT"


def main() -> None:
    candidate_ids = list(GROUND_TRUTH.keys())
    repeats = 5

    observations = run_tournament(
        candidate_ids,
        synthetic_judge,
        repeats=repeats,
    )

    out_dir = Path("/tmp/pairwise_rank_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "observations.jsonl"
    save_observations_jsonl(obs_path, observations)
    print(f"# Wrote {len(observations)} observations to {obs_path}")

    # Re-load (round trip) to demonstrate the file format
    observations = load_observations_jsonl(obs_path)
    print(f"# Reloaded {len(observations)} observations")

    # Direct (model-free) summary
    direct = direct_summary(observations)
    print(f"\n# n_observations = {direct['n_observations']}")
    print(f"# Direct verdict distribution: {direct['per_item']}")

    # BTD fit (default probabilistic model)
    result = fit_btd(observations, item_ids=candidate_ids, draws=1000, tune=1500, chains=4, seed=0)
    summary = summarize_btd(result, observations)

    print(f"\n# verdict_distribution_btd = {summary.get('verdict_distribution_btd', 'n/a')}")

    print("\n# Per-item summary:")
    print(f"{'id':<10}{'theta (true)':>14}{'theta (fit)':>14}{'P(best)':>10}{'E[rank]':>10}")
    for row in summary["per_item"]:
        truth = GROUND_TRUTH[row["id"]]
        print(f"{row['id']:<10}{truth:>+14.2f}{row['theta_mean']:>+14.2f}"
              f"{row['p_best']:>10.3f}{row['expected_rank']:>10.2f}")

    pos = summary["position_effect"]
    print(f"\n# beta_right: {pos['beta_right_mean']:+.3f}  HDI: {pos['beta_right_hdi']}")

    tp = summary.get("tie_parameter")
    if tp:
        print(f"# tie parameter (nu): {tp['nu_mean']:.3f}  HDI: {tp['nu_hdi']}")

    print("\n# Pairwise P(theta_i > theta_j):")
    for key, val in summary["pairwise"].items():
        i, j = key.split(",")
        ii = summary["item_ids"][int(i)]
        jj = summary["item_ids"][int(j)]
        print(f"  P({ii} > {jj}) = {val['p_i_gt_j']:.3f}")

    results_path = out_dir / "fit_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n# Wrote fit summary to {results_path}")


if __name__ == "__main__":
    main()
