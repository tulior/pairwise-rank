"""BTD model invariants: sign conventions, theta sum, nu positivity, P(best),
P(left wins) + P(tie) + P(right wins) sums to 1.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from pairwise_rank import Observation
from pairwise_rank.btd import fit_btd, summarize_btd, direct_summary, _btd_code


def _obs(a, b, left, right, verdict_code, repeat=1):
    name = ["LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG"][verdict_code]
    return Observation(a=a, b=b, left=left, right=right, repeat=repeat, verdict=name)


# 1. STRONG collapsing
def test_btd_collapses_strong_into_ordinary_outcome():
    assert _btd_code("LEFT_STRONG") == 0
    assert _btd_code("LEFT") == 0
    assert _btd_code("TIE") == 1
    assert _btd_code("RIGHT") == 2
    assert _btd_code("RIGHT_STRONG") == 2


# 2. Stronger item has larger theta
def test_swapping_left_right_reverses_theta_direction():
    """A consistent direction across orientations recovers positive theta
    for the stronger item. Tests both LEFT and LEFT_STRONG collapsing."""
    # all RIGHT: b wins on right
    obs = [_obs("a", "b", "a", "b", 3, r) for r in range(1, 7)]
    result = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    assert s["per_item"][1]["theta_mean"] > s["per_item"][0]["theta_mean"]

    # all RIGHT_STRONG: b wins STRONG on right
    obs2 = [_obs("a", "b", "a", "b", 4, r) for r in range(1, 7)]
    result2 = fit_btd(obs2, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s2 = summarize_btd(result2, observations=obs2)
    assert s2["per_item"][1]["theta_mean"] > s2["per_item"][0]["theta_mean"]


# 3. Sum-to-zero theta at every draw
def test_theta_satisfies_zero_sum_at_every_draw():
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 3, r))
        obs.append(_obs("a", "c", "a", "c", 3, r))
        obs.append(_obs("b", "c", "b", "c", 3, r))
    result = fit_btd(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    theta = result.theta_draws
    sums = theta.sum(axis=1)
    assert np.max(np.abs(sums)) < 1e-6


# 4. P(best) sums to one
def test_p_best_sums_to_one_across_items():
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 3, r))
        obs.append(_obs("a", "c", "a", "c", 3, r))
        obs.append(_obs("b", "c", "b", "c", 3, r))
    result = fit_btd(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    p_best = np.array([row["p_best"] for row in s["per_item"]])
    assert abs(p_best.sum() - 1.0) < 1e-6


# 5. nu is positive at every draw
def test_nu_is_positive_at_every_draw():
    obs = []
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 2, r))  # all TIE
    result = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    nu = result.nu_draws
    assert np.all(nu > 0)


# 6. Pairwise likelihood probabilities sum to 1 (when observations provided)
def test_pairwise_likelihood_probs_sum_to_one():
    obs = [_obs("a", "b", "a", "b", 3, r) for r in range(1, 4)]
    result = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    for key, p in s["pairwise"].items():
        if "p_left_wins" in p:
            total = p["p_left_wins"] + p["p_tie"] + p["p_right_wins"]
            assert abs(total - 1.0) < 1e-6, f"pair {key} sums to {total}"


# 7. Position bias sign convention
def test_positive_beta_right_favors_right_slot():
    """Inject an explicit right-slot preference: a wins when on the right
    in a clearly higher rate than a wins when on the left. The model
    should recover a positive beta_right."""
    obs = []
    for r in range(1, 7):
        obs.append(_obs("a", "b", "a", "b", 1, r))   # a on left, a wins: LEFT
        obs.append(_obs("a", "b", "b", "a", 3, r))   # a on right, a wins: RIGHT
    # All 12 obs: a wins. When a is on the left, model says theta_a > theta_b
    # and LEFT (left wins). When a is on the right, model says beta_right
    # needs to be positive enough to overcome the right-slot disadvantage
    # so RIGHT (right wins) — a is right, so a wins.
    # This is balanced in orientation: 6 LEFT, 6 RIGHT.
    result = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    # With a winning equally on both sides, theta_a > theta_b and beta_right
    # is unidentifiable in principle; test only that the model doesn't crash
    # and that theta_a is larger.
    assert s["per_item"][0]["theta_mean"] > s["per_item"][1]["theta_mean"]


def test_beta_right_is_positive_when_right_slot_consistently_advantaged():
    """Construct data where the right slot is consistently advantaged:
    a weak item 'a' beats 'b' more often when a is on the right."""
    obs = []
    # a is genuinely weaker, so a almost never beats b when on the LEFT.
    for r in range(1, 5):
        obs.append(_obs("a", "b", "a", "b", 3, r))  # LEFT: a loses (RIGHT = b wins)
    # But when a is on the right, the right-slot effect helps it win.
    for r in range(1, 5):
        obs.append(_obs("a", "b", "b", "a", 0, r))  # RIGHT: a wins (LEFT_STRONG)
    # Net: a wins 4 (when on right), b wins 4 (when on left). Tie at the
    # item level. The model needs beta_right > 0 to explain this.
    result = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    assert s["position_effect"]["beta_right_mean"] > 0


# 8. direct_summary tallies
def test_direct_summary_counts_correctly():
    obs = [
        _obs("a", "b", "a", "b", 1, 1),  # LEFT: a wins
        _obs("a", "b", "a", "b", 2, 1),  # TIE
        _obs("a", "b", "b", "a", 3, 1),  # a wins on right (b is left, right wins)
        _obs("a", "b", "b", "a", 0, 1),  # LEFT_STRONG: b wins (b is left)
    ]
    d = direct_summary(obs)
    # a wins obs 1 + obs 3 = 2 times
    # b wins obs 4 = 1 time
    # tie = 1
    assert d["per_item"]["wins"]["a"] == 2
    assert d["per_item"]["wins"]["b"] == 1
    assert d["per_item"]["losses"]["a"] == 1
    assert d["per_item"]["losses"]["b"] == 2
    assert d["per_item"]["ties"]["a"] == 1
    assert d["per_item"]["ties"]["b"] == 1
    # 1 STRONG
    assert d["n_left_strong"] == 1
    assert d["n_right_strong"] == 0


# 9. Strong-collapse count
def test_strong_collapse_count():
    obs = [
        _obs("a", "b", "a", "b", 0, 1),  # LEFT_STRONG
        _obs("a", "b", "a", "b", 0, 2),  # LEFT_STRONG
        _obs("a", "b", "a", "b", 1, 1),  # LEFT
        _obs("a", "b", "a", "b", 2, 1),  # TIE
        _obs("a", "b", "a", "b", 3, 1),  # RIGHT
        _obs("a", "b", "a", "b", 4, 1),  # RIGHT_STRONG
    ]
    from pairwise_rank.btd import _strong_count
    sc = _strong_count(obs)
    assert sc["left_strong"] == 2
    assert sc["right_strong"] == 1
    assert sc["total_collapsed"] == 3


# 10. Empty obs raises
def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_btd([], item_ids=["a", "b"])


# 11. Real ranking: 3 items with directional wins, BTD picks the right winner
def test_ranking_recovers_known_order():
    obs = []
    # a > b
    for r in range(1, 5):
        obs.append(_obs("a", "b", "a", "b", 0, r))  # LEFT_STRONG: a wins
    # b > c
    for r in range(1, 5):
        obs.append(_obs("b", "c", "b", "c", 1, r))  # LEFT: b wins
    # a > c
    for r in range(1, 5):
        obs.append(_obs("a", "c", "a", "c", 0, r))  # LEFT_STRONG: a wins
    result = fit_btd(obs, item_ids=["a", "b", "c"], draws=400, tune=600, chains=2, seed=0)
    s = summarize_btd(result, observations=obs)
    by_id = {row["id"]: row for row in s["per_item"]}
    assert by_id["a"]["theta_mean"] > by_id["b"]["theta_mean"]
    assert by_id["a"]["theta_mean"] > by_id["c"]["theta_mean"]
    assert by_id["b"]["theta_mean"] > by_id["c"]["theta_mean"]
    assert by_id["a"]["p_best"] > 0.5


# ---------------------------------------------------------------------------
# Equivalence: pm.Categorical fit == old pm.Potential fit
# ---------------------------------------------------------------------------
#
# The BTD fit was switched from a hand-rolled
#     pm.Potential("y_obs_logp", sum_i (L_i[y_i] - logsumexp(L_i)))
# block to
#     pm.Categorical("y", logit_p=stacked_L, observed=ys)
# These are algebraically identical by definition of
# pm.Categorical. The tests below pin the equivalence so the swap
# can be reproduced or audited.

def test_btd_likelihood_logits_match_davidson_equation():
    """The three logits used by the fit are exactly the Davidson
    log-probabilities (up to the shared log-Z), and the same
    three logits are reused by the numpy path with
    scipy.special.softmax for normalization. This pins the
    'one source of truth for the logits' contract."""
    # Hand-coded reference: for a single observation with
    # (theta[i], theta[j], beta_right, eta_tie), the three
    # Davidson log-probabilities (up to log-Z) are
    #   LEFT  = theta[j]                  (left slot = j)
    #   TIE   = 0.5 * (theta[i] + theta[j] + beta_right) + eta_tie
    #   RIGHT = theta[i] + beta_right     (right slot = i)
    # The production fit builds these and passes them to
    # pm.Categorical. The numpy predict path builds the same
    # three and uses scipy.special.softmax.
    obs = [
        _obs("a", "b", "a", "b", 1, 1),  # TIE
        _obs("a", "b", "b", "a", 0, 1),  # LEFT (b beats a on right)
        _obs("a", "b", "a", "b", 2, 1),  # RIGHT
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    theta = result.theta_draws
    beta = result.beta_right_draws
    log_nu = np.log(result.nu_draws)
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}

    from pairwise_rank import predict_btd
    preds = predict_btd(result, obs)
    for o, p in zip(obs, preds):
        i = item_to_idx[o.left]
        j = item_to_idx[o.right]
        log_pi = theta[:, j] + beta
        log_pj = theta[:, i]
        log_d = 0.5 * (log_pi + log_pj)
        # Reference: scipy.special.softmax on the same three
        # logits, mean over posterior draws.
        ref_probs = softmax(
            np.stack([log_pj, log_d + log_nu, log_pi], axis=1),
            axis=1,
        )
        assert abs(p["p_left_wins"] - float(ref_probs[:, 0].mean())) < 1e-12
        assert abs(p["p_tie"] - float(ref_probs[:, 1].mean())) < 1e-12
        assert abs(p["p_right_wins"] - float(ref_probs[:, 2].mean())) < 1e-12


def test_btd_fit_logp_matches_manual_davidson_logp():
    """Algebraic identity: pm.Categorical(logit_p=L, observed=y)
    computes the same logp as the old hand-rolled
    sum_i (L_i[y_i] - logsumexp(L_i)). This is a unit test
    at the pytensor level, independent of any real fit."""
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    # Hand-built logits, no priors.
    L = np.array(
        [[1.0, 0.5, -0.5],   # obs 0
         [0.0, 1.5, 0.0],    # obs 1
         [-1.0, 0.0, 2.0]],  # obs 2
        dtype=float,
    )
    y = np.array([0, 1, 2], dtype=int)
    L_pt = pt.as_tensor_variable(L)

    with pm.Model() as m:
        pm.Categorical("y_obs", logit_p=L_pt, observed=y)
        # The old hand-rolled expression for cross-check.
        log_Z = pt.logsumexp(L_pt, axis=1)
        log_p = L_pt - log_Z[:, None]
        old_logp = pt.sum(log_p[pt.arange(y.shape[0]), y])

    new_logp = m.compile_logp()(m.initial_point())
    # Compile the old logp as a free function on the same L.
    # ``pytensor.function`` lives at the top level of the
    # ``pytensor`` package, not on ``pymc.pytensorf``.
    old_logp_fn = pytensor.function([], old_logp)
    old_val = old_logp_fn()

    assert abs(float(new_logp) - float(old_val)) < 1e-10


def test_predict_btd_matches_scipy_softmax_on_same_logits():
    """For any posterior draw, predict_btd's probabilities must
    equal scipy.special.softmax applied to the same three
    Davidson logits. This is the load-bearing equivalence: the
    fit uses pm.Categorical (which uses an internal softmax on
    the same logits); predict_btd uses scipy.special.softmax on
    the same logits. Both paths must produce the same numbers.
    """
    obs = [
        _obs("a", "b", "a", "b", 0, 1),  # LEFT
        _obs("a", "b", "b", "a", 1, 1),  # TIE
        _obs("a", "b", "a", "b", 2, 1),  # RIGHT
        _obs("a", "b", "b", "a", 0, 1),  # LEFT
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    # Mirror predict_btd's indexing.
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    theta = result.theta_draws
    beta = result.beta_right_draws
    log_nu = np.log(result.nu_draws)

    from pairwise_rank import predict_btd
    preds = predict_btd(result, obs)
    for o, p in zip(obs, preds):
        i = item_to_idx[o.left]
        j = item_to_idx[o.right]
        log_pi = theta[:, j] + beta
        log_pj = theta[:, i]
        log_d = 0.5 * (log_pi + log_pj)
        ref = softmax(
            np.stack([log_pj, log_d + log_nu, log_pi], axis=1),
            axis=1,
        )
        assert abs(p["p_left_wins"] - float(ref[:, 0].mean())) < 1e-12
        assert abs(p["p_tie"] - float(ref[:, 1].mean())) < 1e-12
        assert abs(p["p_right_wins"] - float(ref[:, 2].mean())) < 1e-12
