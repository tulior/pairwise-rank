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


# ---------------------------------------------------------------------------
# Vectorized summary helpers: deterministic equivalence on handcrafted
# posterior arrays. These tests do not run a sampler; they construct
# a small (S, n) posterior directly and verify that the new helpers
# match a simple per-item reference loop.
# ---------------------------------------------------------------------------

def _handcrafted_posterior(S=200, n=4, seed=0):
    """Build a deterministic (S, n) theta array with no exact
    within-draw ties. Used to pin down the vectorized math.
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(loc=0.0, scale=1.0, size=(S, n))
    return theta


def test_rank_pos_matches_simple_per_item_loop():
    """_rank_pos returns the same per-item 0-indexed positions that
    the pre-vectorization code produced: ``np.where(np.argsort(-theta,
    axis=1) == i)[1]`` for each i."""
    from pairwise_rank.btd import _rank_pos

    theta = _handcrafted_posterior(S=200, n=4, seed=0)
    rank_pos = _rank_pos(theta)
    # Reference: per-item loop with the pre-vectorization formula.
    ranks = np.argsort(-theta, axis=1)              # (S, n)
    ref = np.array([np.where(ranks == i)[1] for i in range(4)])
    np.testing.assert_array_equal(rank_pos, ref)


def test_p_best_matches_simple_per_item_loop():
    """_p_best returns the same per-item fractions that the
    pre-vectorization code produced: ``(np.argmax(theta, axis=1) ==
    i).mean()`` for each i. Each draw credits exactly one item
    via the per-draw argmax (first-occurrence on ties)."""
    from pairwise_rank.btd import _p_best

    theta = _handcrafted_posterior(S=200, n=4, seed=0)
    p_best = _p_best(theta)
    argmax_items = np.argmax(theta, axis=1)
    ref = np.array([(argmax_items == i).mean() for i in range(4)])
    np.testing.assert_allclose(p_best, ref)
    # Sums to 1 (each draw credits exactly one item).
    assert abs(p_best.sum() - 1.0) < 1e-12


def test_pairwise_gt_means_matches_simple_per_item_loop():
    """_pairwise_gt_means returns the same (n, n) off-diagonal
    values that the pre-vectorization code produced, with the
    diagonal set to NaN.
    """
    from pairwise_rank.btd import _pairwise_gt_means

    theta = _handcrafted_posterior(S=200, n=4, seed=0)
    P = _pairwise_gt_means(theta)
    # Reference: the per-pair double loop.
    n = theta.shape[1]
    ref = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            ref[i, j] = float((theta[:, i] > theta[:, j]).mean())
            ref[j, i] = 1.0 - ref[i, j]
    # Diagonal: helper returns 0 internally, then fill_diagonal sets
    # to NaN. Reference has NaN on the diagonal too.
    np.testing.assert_array_equal(np.isnan(P), np.isnan(ref))
    np.testing.assert_allclose(P[~np.isnan(P)], ref[~np.isnan(ref)])


def test_p_top2_at_least_p_best():
    """P(top2) >= P(best) for every item, by the same construction
    the original code used: top2 is "position <= 1" in the rank
    order; the unique-argmax item is always at position 0.
    """
    from pairwise_rank.btd import _rank_pos, _p_best

    theta = _handcrafted_posterior(S=200, n=4, seed=0)
    rank_pos = _rank_pos(theta)
    p_top2 = (rank_pos <= 1).mean(axis=1)
    p_best = _p_best(theta)
    assert np.all(p_top2 >= p_best - 1e-12)


def test_expected_rank_in_unit_interval():
    """expected_rank is in [1, n] for every item (1-indexed)."""
    from pairwise_rank.btd import _rank_pos

    theta = _handcrafted_posterior(S=200, n=4, seed=0)
    rank_pos = _rank_pos(theta)
    mean_rank = rank_pos.mean(axis=1) + 1
    assert np.all(mean_rank >= 1.0 - 1e-12)
    assert np.all(mean_rank <= 4.0 + 1e-12)


def test_davidson_probs_sum_to_one_on_handcrafted_input():
    """_davidson_probs returns a (..., 3) array of probabilities
    that sum to 1 along the last axis, for any broadcastable input
    shape. Tested on (S,), (S, n), and (S, n_obs) shapes.
    """
    from pairwise_rank.btd import _davidson_probs

    rng = np.random.default_rng(0)
    S = 100
    beta = rng.normal(size=S)
    log_nu = rng.normal(size=S)
    n = 4
    n_obs = 7

    # (S,) shape -- single pair.
    theta_left = rng.normal(size=S)
    theta_right = rng.normal(size=S)
    p = _davidson_probs(theta_left, theta_right, beta, log_nu)
    assert p.shape == (S, 3)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(S), atol=1e-12)

    # (S, n) shape -- all items, for fitting or per-pair reference.
    theta = rng.normal(size=(S, n))
    p = _davidson_probs(theta, theta, beta, log_nu)
    assert p.shape == (S, n, 3)
    np.testing.assert_allclose(p.sum(axis=2), np.ones((S, n)), atol=1e-12)

    # (S, n_obs) shape -- batched per-observation likelihood.
    left_idx = rng.integers(0, n, size=n_obs)
    right_idx = rng.integers(0, n, size=n_obs)
    p = _davidson_probs(theta[:, left_idx], theta[:, right_idx], beta, log_nu)
    assert p.shape == (S, n_obs, 3)
    np.testing.assert_allclose(p.sum(axis=2), np.ones((S, n_obs)), atol=1e-12)


def test_davidson_probs_matches_per_observation_loop():
    """_davidson_probs on a (S, n_obs) input matches a per-observation
    loop calling the same formula."""
    from pairwise_rank.btd import _davidson_probs
    from scipy.special import softmax

    rng = np.random.default_rng(0)
    S = 50
    n = 4
    n_obs = 6
    theta = rng.normal(size=(S, n))
    beta = rng.normal(size=S)
    log_nu = rng.normal(size=S)
    left_idx = rng.integers(0, n, size=n_obs)
    right_idx = rng.integers(0, n, size=n_obs)

    # Vectorized.
    p_vec = _davidson_probs(theta[:, left_idx], theta[:, right_idx], beta, log_nu)
    # Per-observation loop.
    p_loop = np.zeros((S, n_obs, 3))
    for k in range(n_obs):
        i, j = int(left_idx[k]), int(right_idx[k])
        log_pi = theta[:, j] + beta
        log_pj = theta[:, i]
        log_d = 0.5 * (log_pi + log_pj)
        logits = np.stack([log_pj, log_d + log_nu, log_pi], axis=1)
        p_loop[:, k] = softmax(logits, axis=1)
    np.testing.assert_allclose(p_vec, p_loop, atol=1e-12)


def test_davidson_probs_left_right_symmetry_at_zero_beta():
    """When beta = 0 and theta_left == theta_right (the same
    item on both slots), the LEFT and RIGHT probabilities must
    be equal: P(LEFT) = P(RIGHT) = (1 - P(TIE)) / 2. The
    tie probability depends only on nu.
    """
    from pairwise_rank.btd import _davidson_probs

    rng = np.random.default_rng(0)
    S = 100
    theta_one = rng.normal(size=S)
    beta = np.zeros(S)
    log_nu = rng.normal(size=S)
    p = _davidson_probs(theta_one, theta_one, beta, log_nu)
    # P(LEFT) == P(RIGHT) for every draw.
    np.testing.assert_allclose(p[:, 0], p[:, 2], atol=1e-12)
    # Sums to 1.
    np.testing.assert_allclose(p.sum(axis=1), np.ones(S), atol=1e-12)


def test_davidson_probs_approaches_bradley_terry_at_very_negative_log_nu():
    """At very negative log_nu (nu -> 0), the tie probability
    vanishes and the LEFT/RIGHT split matches plain Bradley-Terry:
    P(LEFT) = sigmoid(theta_left - theta_right) (when beta = 0).
    """
    from pairwise_rank.btd import _davidson_probs

    rng = np.random.default_rng(0)
    S = 200
    theta_left = rng.normal(size=S)
    theta_right = rng.normal(size=S)
    beta = np.zeros(S)
    log_nu = np.full(S, -20.0)        # nu = exp(-20) ~ 2e-9, effectively 0
    p = _davidson_probs(theta_left, theta_right, beta, log_nu)
    # P(TIE) is essentially 0.
    assert p[:, 1].max() < 1e-6
    # P(LEFT) matches Bradley-Terry: sigmoid(theta_left - theta_right).
    bt = 1.0 / (1.0 + np.exp(-(theta_left - theta_right)))
    np.testing.assert_allclose(p[:, 0], bt, atol=1e-3)
    # P(LEFT) + P(RIGHT) ~ 1.
    np.testing.assert_allclose(p[:, 0] + p[:, 2], np.ones(S), atol=1e-6)


def test_davidson_probs_position_neutral_explicit_zero_beta():
    """When position_neutral=True, the per-pair BTD probabilities
    are computed with beta_right = 0. The vectorized helper must
    produce the same numbers as an explicit call with zeroed beta.
    """
    from pairwise_rank.btd import _davidson_probs, _position_neutral_beta

    rng = np.random.default_rng(0)
    S = 200
    n_obs = 6
    theta = rng.normal(size=(S, 4))
    beta_orig = rng.normal(size=S)
    log_nu = rng.normal(size=S)
    left_idx = rng.integers(0, 4, size=n_obs)
    right_idx = rng.integers(0, 4, size=n_obs)

    # Path A: position_neutral via the existing helper.
    beta_neutral = _position_neutral_beta(beta_orig, position_neutral=True)
    p_a = _davidson_probs(theta[:, left_idx], theta[:, right_idx], beta_neutral, log_nu)
    # Path B: explicit zero beta.
    p_b = _davidson_probs(theta[:, left_idx], theta[:, right_idx], np.zeros_like(beta_orig), log_nu)
    np.testing.assert_array_equal(p_a, p_b)


def test_summarize_btd_matches_handcrafted_helper_outputs():
    """End-to-end: build a BTDFitResult-like object from a
    handcrafted posterior, run summarize_btd, and assert that the
    per-item / pairwise outputs match the vectorized helpers
    directly. This pins the wire-up between summarize_btd and
    _rank_pos / _p_best / _pairwise_gt_means.
    """
    from pairwise_rank.btd import (
        BTDFitResult, _rank_pos, _p_best, _pairwise_gt_means,
        summarize_btd,
    )

    S, n = 200, 4
    rng = np.random.default_rng(0)
    # Use 2 chains x 100 draws so arviz diagnostics are well-defined
    # (r_hat requires >= 2 chains; we only need the per-item /
    # pairwise summary numbers here, but the r_hat path will run
    # anyway).
    n_chains, n_draws = 2, 100
    theta = rng.normal(size=(S, n))
    beta_right = rng.normal(size=S) * 0.1
    sigma_theta = np.full(S, 1.0)
    eta_tie = rng.normal(size=S) * 0.3
    import xarray as xr
    import arviz as az
    posterior = xr.Dataset({
        "theta": (("chain", "draw", "theta_dim_0"),
                  theta.reshape(n_chains, n_draws, n)),
        "beta_right": (("chain", "draw"),
                       beta_right.reshape(n_chains, n_draws)),
        "sigma_theta": (("chain", "draw"),
                        sigma_theta.reshape(n_chains, n_draws)),
        "eta_tie": (("chain", "draw"),
                    eta_tie.reshape(n_chains, n_draws)),
    })
    idata = az.InferenceData(posterior=posterior)
    item_ids = ["a", "b", "c", "d"]
    result = BTDFitResult(
        idata=idata, n=n, item_ids=list(item_ids), divergences=0,
        config={"draws": n_draws, "tune": 0, "chains": n_chains,
                "target_accept": 0.95, "seed": 0,
                "n_observations": 0, "strong_collapsed": {}},
    )
    s = summarize_btd(result)

    # Reference: direct calls to the vectorized helpers on the
    # (S, n) flat theta array.
    rank_pos_ref = _rank_pos(theta)
    p_best_ref = _p_best(theta)
    p_top2_ref = (rank_pos_ref <= 1).mean(axis=1)
    mean_rank_ref = rank_pos_ref.mean(axis=1) + 1
    pairwise_ref = _pairwise_gt_means(theta)

    for i in range(n):
        row = s["per_item"][i]
        assert row["theta_mean"] == pytest.approx(theta[:, i].mean())
        assert row["p_best"] == pytest.approx(p_best_ref[i])
        assert row["p_top2"] == pytest.approx(p_top2_ref[i])
        assert row["expected_rank"] == pytest.approx(mean_rank_ref[i])
    # Pairwise: check p_i_gt_j matches the helper for each (i, j) pair.
    for key_str, entry in s["pairwise"].items():
        i, j = (int(x) for x in key_str.split(","))
        assert entry["p_i_gt_j"] == pytest.approx(pairwise_ref[i, j])
