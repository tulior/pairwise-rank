"""Synthetic example: full pipeline with a deterministic ground-truth judge.

Run with:
    python examples/synthetic.py

Generates a small tournament over 4 fake candidates using a
deterministic judge, writes observations to JSONL, fits the default
model, and prints a summary. The judge here is a stand-in: replace
it with your own callable for real use.
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
    fit,
    summarize,
    posterior_predictive_check,
)


GROUND_TRUTH = {
    "alpha": 1.5,
    "beta": 0.5,
    "gamma": -0.5,
    "delta": -1.5,
}

# Cutpoints for the synthetic judge: -1.5, -0.5, +0.5, +1.5
CP = [-1.5, -0.5, 0.5, 1.5]


def synthetic_judge(left_id: str, right_id: str) -> str:
    """Deterministic stand-in judge. The strength difference drives eta;
    small noise is added so the synthetic data has some disagreement
    across repeats. Replace with your own callable for real use.
    """
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

    result = fit(observations, item_ids=candidate_ids, draws=1000, tune=1500, chains=4, seed=0)
    summary = summarize(result, observations)

    print(f"\n# n_observations = {summary['n_observations']}")
    print(f"# verdict_distribution = {summary['verdict_distribution']}")

    print("\n# Per-item summary:")
    print(f"{'id':<10}{'theta (true)':>14}{'theta (fit)':>14}{'P(best)':>10}{'E[rank]':>10}")
    for row in summary["per_item"]:
        truth = GROUND_TRUTH[row["id"]]
        print(f"{row['id']:<10}{truth:>+14.2f}{row['theta_mean']:>+14.2f}"
              f"{row['p_best']:>10.3f}{row['expected_rank']:>10.2f}")

    pos = summary["position_effect"]
    print(f"\n# beta_right: {pos['beta_right_mean']:+.3f}  HDI: {pos['beta_right_hdi']}")

    print("\n# Pairwise P(theta_i > theta_j):")
    for key, val in summary["pairwise"].items():
        i, j = key.split(",")
        # Map indices back to ids for the print
        ii = summary["item_ids"][int(i)]
        jj = summary["item_ids"][int(j)]
        print(f"  P({ii} > {jj}) = {val['p_i_gt_j']:.3f}")

    ppc = posterior_predictive_check(result, observations, n_ppc=500, seed=0)
    print(f"\n# PPC repeat-agreement: observed={ppc['observed']:.3f}, "
          f"ppc mean={ppc['ppc_mean']:.3f}, p_ppc_ge_observed={ppc['p_ppc_ge_observed']:.3f}")

    results_path = out_dir / "fit_summary.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n# Wrote fit summary to {results_path}")


if __name__ == "__main__":
    main()
