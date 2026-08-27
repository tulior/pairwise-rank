"""Model invariants: sign conventions, theta sum, cutpoints, P(best),
position-neutral prediction.
"""
from __future__ import annotations

import numpy as np
import pytest

from pairwise_rank import (
    Observation,
    fit,
    summarize,
    posterior_predictive_check,
)


def _obs(a, b, left, right, verdict_code, repeat=1):
    name = ["LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG"][verdict_code]
    return Observation(a=a, b=b, left=left, right=right, repeat=repeat, verdict=name)


# 6. left/right sign reversal
def test_swapping_left_right_reverses_theta_direction():
    """A consistent direction across orientations recovers positive theta
    for the stronger item in both cases."""
    obs = [_obs("a", "b", "a", "b", 3, r) for r in range(1, 7)]  # all RIGHT (b wins on right)
    result = fit(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize(result)
    assert s["per_item"][1]["theta_mean"] > s["per_item"][0]["theta_mean"]  # b > a

    obs2 = [_obs("a", "b", "b", "a", 3, r) for r in range(1, 7)]  # a wins when on right
    result2 = fit(obs2, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s2 = summarize(result2)
    assert s2["per_item"][0]["theta_mean"] > s2["per_item"][1]["theta_mean"]  # a > b


# 7. beta sign convention
def test_positive_beta_right_favors_right_slot():
    """All-right observations with no inherent preference for either item
    produce positive beta_right."""
    obs = [_obs("a", "b", "a", "b", 4, r) for r in range(1, 5)]
    result = fit(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize(result)
    assert s["position_effect"]["beta_right_mean"] > 0


# 8. centered ordered cutpoints
def test_cutpoints_are_ordered_and_centered():
    """Cutpoints must be strictly ordered and sum to zero at every draw."""
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 1, r))   # LEFT
        obs.append(_obs("a", "c", "a", "b", 3, r))   # RIGHT (relative strength)
        obs.append(_obs("b", "c", "b", "c", 2, r))   # TIE
    result = fit(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    cps = result.cutpoint_draws  # (S, 4)
    # Strictly ordered
    for s_draw in cps:
        for k in range(3):
            assert s_draw[k] < s_draw[k + 1], f"cutpoints not ordered: {s_draw}"
    # Sum to zero
    sums = cps.sum(axis=1)
    assert np.max(np.abs(sums)) < 1e-6


# 9. zero-sum theta
def test_theta_satisfies_zero_sum_at_every_draw():
    """ZeroSumNormal enforces sum-to-zero at every draw."""
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 3, r))
        obs.append(_obs("a", "c", "a", "c", 3, r))
        obs.append(_obs("b", "c", "b", "c", 3, r))
    result = fit(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    theta = result.theta_draws
    sums = theta.sum(axis=1)
    assert np.max(np.abs(sums)) < 1e-6


# 10. joint P(best)
def test_p_best_sums_to_one_across_items():
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 3, r))
        obs.append(_obs("a", "c", "a", "c", 3, r))
        obs.append(_obs("b", "c", "b", "c", 3, r))
    result = fit(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    s = summarize(result)
    p_best = np.array([row["p_best"] for row in s["per_item"]])
    assert abs(p_best.sum() - 1.0) < 1e-6


# 11. position-neutral prediction
def test_balanced_data_drives_beta_right_toward_zero():
    """When the data is balanced (50% RIGHT in L_first, 50% RIGHT in R_first),
    the posterior of beta_right should center near zero.
    """
    obs = []
    for r in range(1, 5):
        # L_first: half LEFT, half RIGHT
        if r % 2 == 0:
            obs.append(_obs("a", "b", "a", "b", 1, r))  # LEFT
        else:
            obs.append(_obs("a", "b", "a", "b", 3, r))  # RIGHT
        # R_first: opposite pattern to balance
        if r % 2 == 0:
            obs.append(_obs("a", "b", "b", "a", 3, r))  # RIGHT
        else:
            obs.append(_obs("a", "b", "b", "a", 1, r))  # LEFT
    result = fit(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize(result)
    # Posterior mean of beta_right should be near zero
    assert abs(s["position_effect"]["beta_right_mean"]) < 0.3


# bonus: empty obs raises
def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit([], item_ids=["a", "b"])


# bonus: ppc returns expected fields
def test_ppc_returns_expected_fields():
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 3, r))
        obs.append(_obs("a", "c", "a", "c", 3, r))
    result = fit(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    ppc = posterior_predictive_check(result, obs, n_ppc=200, seed=0)
    for k in ("observed", "ppc_mean", "ppc_hdi_90", "p_ppc_ge_observed"):
        assert k in ppc
    assert 0.0 <= ppc["p_ppc_ge_observed"] <= 1.0
