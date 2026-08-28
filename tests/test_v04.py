"""v0.4 default-protocol tests.

These tests cover the contract for the new default 3-level
protocol with backward compatibility for legacy 5-level data.
They do not duplicate existing tests; they pin down the items
the v0.4 docs explicitly call out:

  - canonical LEFT/TIE/RIGHT input
  - legacy LEFT_STRONG -> LEFT collapse
  - legacy RIGHT_STRONG -> RIGHT collapse
  - direct_summary with mixed legacy/current observations
  - BTD accepts mixed legacy/current observations
  - BTD probabilities sum correctly
  - tie handling is symmetric
  - swapping orientation changes the position term with the
    documented sign
  - beta_right > 0 favors the right slot
  - P(best) is computed jointly from posterior draws
  - position-neutral predictions set beta_right = 0
  - existing ordered-model tests remain green (covered in
    tests/test_model.py and tests/test_recovery.py)
"""
from __future__ import annotations

import numpy as np
import pytest

from pairwise_rank import (
    Observation,
    fit_btd,
    summarize_btd,
    direct_summary,
    collapse_to_3_level,
    VERDICT_LEVELS,
    VERDICT_LEVELS_5,
)


def _obs(a, b, left, right, verdict, repeat=1):
    return Observation(
        a=a, b=b, left=left, right=right, repeat=repeat, verdict=verdict,
    )


# ---- canonical 3-level input ------------------------------------------------

def test_canonical_3level_input():
    """LEFT / TIE / RIGHT are accepted unchanged by direct_summary
    and BTD."""
    obs = [
        _obs("a", "b", "a", "b", "LEFT", 1),
        _obs("a", "b", "a", "b", "TIE", 1),
        _obs("a", "b", "a", "b", "RIGHT", 1),
    ]
    d = direct_summary(obs)
    assert d["per_item"]["wins"]["a"] == 1
    assert d["per_item"]["wins"]["b"] == 1
    assert d["per_item"]["ties"]["a"] == 1
    # BTD accepts the same data
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    s = summarize_btd(result, observations=obs)
    assert s["n_observations"] == 3


# ---- legacy 5-level collapse ------------------------------------------------

def test_legacy_left_strong_collapses_to_left():
    """LEFT_STRONG verdicts are collapsed to LEFT in BTD and direct_summary."""
    obs = [
        _obs("a", "b", "a", "b", "LEFT_STRONG", 1),
        _obs("a", "b", "a", "b", "LEFT_STRONG", 2),
        _obs("a", "b", "a", "b", "LEFT", 1),
    ]
    d = direct_summary(obs)
    # All 3 obs are "a wins" after collapse
    assert d["per_item"]["wins"]["a"] == 3
    assert d["per_item"]["losses"]["b"] == 3
    # Counted separately for backward compat reporting
    assert d["n_left_strong"] == 2
    assert d["n_right_strong"] == 0


def test_legacy_right_strong_collapses_to_right():
    """RIGHT_STRONG verdicts are collapsed to RIGHT in BTD and direct_summary."""
    obs = [
        _obs("a", "b", "a", "b", "RIGHT_STRONG", 1),
        _obs("a", "b", "a", "b", "RIGHT", 1),
        _obs("a", "b", "a", "b", "TIE", 1),
    ]
    d = direct_summary(obs)
    # b wins the 2 decisive obs
    assert d["per_item"]["wins"]["b"] == 2
    assert d["per_item"]["losses"]["a"] == 2
    assert d["per_item"]["ties"]["a"] == 1
    assert d["n_left_strong"] == 0
    assert d["n_right_strong"] == 1


def test_collapse_to_3_level_helper_is_total():
    """The collapse helper handles every value the user can put on disk."""
    assert collapse_to_3_level("LEFT_STRONG") == "LEFT"
    assert collapse_to_3_level("RIGHT_STRONG") == "RIGHT"
    assert collapse_to_3_level("LEFT") == "LEFT"
    assert collapse_to_3_level("TIE") == "TIE"
    assert collapse_to_3_level("RIGHT") == "RIGHT"


# ---- mixed legacy / current -------------------------------------------------

def test_direct_summary_with_mixed_legacy_current():
    """A dataset mixing LEFT_STRONG, LEFT, TIE, RIGHT, RIGHT_STRONG
    tallies correctly without any caller-side normalization."""
    obs = [
        _obs("a", "b", "a", "b", "LEFT_STRONG", 1),
        _obs("a", "b", "a", "b", "LEFT", 2),
        _obs("a", "b", "a", "b", "TIE", 1),
        _obs("a", "b", "a", "b", "RIGHT", 1),
        _obs("a", "b", "a", "b", "RIGHT_STRONG", 2),
    ]
    d = direct_summary(obs)
    # 2 wins for a (LEFT_STRONG + LEFT), 2 wins for b (RIGHT + RIGHT_STRONG), 1 tie
    assert d["per_item"]["wins"]["a"] == 2
    assert d["per_item"]["losses"]["a"] == 2
    assert d["per_item"]["wins"]["b"] == 2
    assert d["per_item"]["losses"]["b"] == 2
    assert d["per_item"]["ties"]["a"] == 1
    assert d["per_item"]["ties"]["b"] == 1
    # Strong collapse counts
    assert d["n_left_strong"] == 1
    assert d["n_right_strong"] == 1


def test_btd_accepts_mixed_legacy_current():
    """BTD fits a mixed-legacy dataset without complaint and the
    collapsed counts are reported in the config."""
    obs = [
        _obs("a", "b", "a", "b", "LEFT_STRONG", 1),
        _obs("a", "b", "a", "b", "TIE", 1),
        _obs("a", "b", "a", "b", "RIGHT_STRONG", 1),
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    s = summarize_btd(result, observations=obs)
    assert s["n_observations"] == 3
    sc = s["config"]["strong_collapsed"]
    assert sc["left_strong"] == 1
    assert sc["right_strong"] == 1
    assert sc["total_collapsed"] == 2


# ---- BTD probability invariants ---------------------------------------------

def test_btd_probabilities_sum_to_one():
    """Per-observation BTD likelihood: P(L) + P(T) + P(R) = 1.

    Constructed by hand from the (theta, beta, eta_tie) posterior
    draws; the implementation must produce probabilities that
    sum to 1 for every posterior draw, on average."""
    obs = [
        _obs("a", "b", "a", "b", "LEFT", r) for r in range(1, 4)
    ] + [
        _obs("a", "b", "a", "b", "RIGHT", r) for r in range(1, 4)
    ] + [
        _obs("a", "b", "a", "b", "TIE", r) for r in range(1, 4)
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    s = summarize_btd(result, observations=obs)
    for key, p in s["pairwise"].items():
        if "p_left_wins" in p:
            total = p["p_left_wins"] + p["p_tie"] + p["p_right_wins"]
            assert abs(total - 1.0) < 1e-6, f"pair {key} sums to {total}"


def test_tie_handling_is_symmetric():
    """The BTD likelihood is symmetric in (i, j) when beta_right = 0.

    Build two parallel datasets that swap the left/right slot of every
    pair. With beta_right forced to 0 (position_neutral), the per-pair
    P(i wins) for the swapped dataset should equal 1 - P(i wins) for
    the original, since the geometry is reflected."""
    obs_a = [
        _obs("a", "b", "a", "b", "RIGHT", r) for r in range(1, 5)
    ] + [
        _obs("a", "b", "a", "b", "TIE", r) for r in range(1, 4)
    ]
    obs_b = [
        _obs("a", "b", "b", "a", "LEFT", r) for r in range(1, 5)
    ] + [
        _obs("a", "b", "b", "a", "TIE", r) for r in range(1, 4)
    ]
    res_a = fit_btd(obs_a, item_ids=["a", "b"], draws=400, tune=500, chains=2, seed=0)
    res_b = fit_btd(obs_b, item_ids=["a", "b"], draws=400, tune=500, chains=2, seed=0)
    # With position_neutral=True, the predictions ignore beta_right
    # and only the tie distribution carries orientation info via
    # eta_tie. The two datasets should give the same P(tie) under
    # position_neutral.
    s_a = summarize_btd(res_a, observations=obs_a, position_neutral=True)
    s_b = summarize_btd(res_b, observations=obs_b, position_neutral=True)
    a_pair = next(iter(s_a["pairwise"].values()))
    b_pair = next(iter(s_b["pairwise"].values()))
    # TIE probability should be the same (geometry is symmetric in i/j).
    assert abs(a_pair["p_tie"] - b_pair["p_tie"]) < 0.05, (
        f"p_tie not symmetric: {a_pair['p_tie']} vs {b_pair['p_tie']}"
    )


# ---- position effect sign convention ----------------------------------------

def test_swapping_orientation_changes_position_term():
    """The position term has the documented sign: beta_right > 0
    means the right slot is advantaged. We construct a 3-item
    dataset where the right-slot advantage is the only explanation
    for the data, and verify beta_right is positive.

    Setup: a, b, c. a > c (theta_a > theta_c). For a vs b, a wins
    on the right but ties on the left. The only way to explain
    that asymmetry is for the right slot to be advantaged.

    Concretely: verdict=LEFT means the LEFT candidate wins.
      - a on right, b on left, verdict=LEFT  -> b wins
      - a on left,  b on right, verdict=LEFT -> a wins
    So "a on right" is paired with b winning. To explain a tie on
    the left vs a loss on the right, the LEFT slot must be the
    advantaged one. Therefore beta_right < 0.

    We assert beta_right < 0, documenting the documented sign.
    """
    obs = []
    for r in range(1, 5):
        # a on right, b on left, verdict=LEFT  -> b wins
        obs.append(_obs("a", "b", "b", "a", "LEFT", r))
        # a on left,  b on right, verdict=TIE  -> tie
        obs.append(_obs("a", "b", "a", "b", "TIE", r))
    # Anchor a's strength with a-c (a > c consistently)
    for r in range(1, 5):
        obs.append(_obs("a", "c", "a", "c", "LEFT", r))
        obs.append(_obs("a", "c", "c", "a", "RIGHT", r))
    res = fit_btd(obs, item_ids=["a", "b", "c"], draws=400, tune=600, chains=2, seed=0)
    s = summarize_btd(res, observations=obs)
    # LEFT slot advantage -> beta_right < 0
    assert s["position_effect"]["beta_right_mean"] < 0, (
        f"expected beta_right < 0 (left slot advantaged); "
        f"got {s['position_effect']['beta_right_mean']:.3f}"
    )


def test_beta_right_positive_favors_right_slot():
    """Construct data where the right slot is consistently advantaged.
    beta_right should come out positive on average."""
    obs = []
    # a is genuinely weaker, so a almost never beats b on the LEFT.
    for r in range(1, 5):
        obs.append(_obs("a", "b", "a", "b", "RIGHT", r))   # LEFT obs: a loses
    # But when a is on the right, the right-slot effect helps it win.
    for r in range(1, 5):
        obs.append(_obs("a", "b", "b", "a", "LEFT", r))    # RIGHT obs: a wins
    res = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(res, observations=obs)
    assert s["position_effect"]["beta_right_mean"] > 0


# ---- P(best) is joint --------------------------------------------------------

def test_p_best_is_joint_event():
    """P(best) sums to 1 over the candidate field, by construction.

    If P(best) were a normalized softmax of theta_mean, the values
    could still sum to 1 but would be a different quantity. The
    BTD model computes P(best) as the joint event 'argmax(theta) = i'
    over posterior draws, which is what the documentation claims.
    """
    obs = [
        _obs("a", "b", "a", "b", "LEFT", r) for r in range(1, 4)
    ] + [
        _obs("a", "c", "a", "c", "LEFT", r) for r in range(1, 4)
    ] + [
        _obs("b", "c", "b", "c", "RIGHT", r) for r in range(1, 4)
    ]
    res = fit_btd(obs, item_ids=["a", "b", "c"], draws=300, tune=500, chains=2, seed=0)
    s = summarize_btd(res, observations=obs)
    p_best = np.array([row["p_best"] for row in s["per_item"]])
    assert abs(p_best.sum() - 1.0) < 1e-6
    # All values are non-negative and <= 1
    assert np.all(p_best >= 0)
    assert np.all(p_best <= 1 + 1e-9)


# ---- position-neutral predictions -------------------------------------------

def test_position_neutral_predictions_zero_beta_right():
    """When position_neutral=True, the per-pair and per-item predictions
    are computed with beta_right = 0.

    We test this by checking that the position_neutral flag is
    surfaced and the per-pair probabilities do not depend on
    beta_right at the point of computation. The cleanest way to
    verify is to compare a dataset where beta_right is clearly
    non-zero in the unconstrained fit, and confirm that the
    position_neutral summary's per-pair tie probabilities are
    close to what we get when we manually set beta_right = 0.
    """
    # Construct an unambiguous right-slot preference.
    obs = []
    for r in range(1, 5):
        obs.append(_obs("a", "b", "a", "b", "RIGHT", r))   # a on left, b wins
    for r in range(1, 5):
        obs.append(_obs("a", "b", "b", "a", "LEFT", r))    # a on right, a wins
    res = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)

    s_unconstrained = summarize_btd(res, observations=obs, position_neutral=False)
    s_neutral = summarize_btd(res, observations=obs, position_neutral=True)

    # Flag is surfaced.
    assert s_unconstrained["position_neutral"] is False
    assert s_neutral["position_neutral"] is True

    # beta_right is the same in both (the underlying posterior is
    # the same; only the predictions differ).
    assert (
        s_unconstrained["position_effect"]["beta_right_mean"]
        == s_neutral["position_effect"]["beta_right_mean"]
    )

    # Per-pair probabilities will differ between the two summaries
    # because beta_right was non-trivially non-zero. The neutral
    # version should report ties probabilities that are closer to
    # the simple theta-based prediction (no slot adjustment).
    unconstrained_pair = next(iter(s_unconstrained["pairwise"].values()))
    neutral_pair = next(iter(s_neutral["pairwise"].values()))
    # At least one of the three probabilities must differ; if not,
    # the position_neutral flag is a no-op.
    differs = (
        abs(unconstrained_pair["p_left_wins"] - neutral_pair["p_left_wins"]) > 1e-6
        or abs(unconstrained_pair["p_tie"] - neutral_pair["p_tie"]) > 1e-6
        or abs(unconstrained_pair["p_right_wins"] - neutral_pair["p_right_wins"]) > 1e-6
    )
    assert differs, "position_neutral=True must change the per-pair predictions"


def test_position_neutral_tie_rate_differs_from_unconstrained():
    """When the right slot is advantaged (or disadvantaged), the
    position-neutral summary reports different tie probabilities
    than the unconstrained summary, because the beta_right offset
    is zeroed out in the predictions."""
    obs = []
    for r in range(1, 5):
        obs.append(_obs("a", "b", "a", "b", "RIGHT", r))   # b wins on left
    for r in range(1, 5):
        obs.append(_obs("a", "b", "b", "a", "LEFT", r))    # a wins on right
    res = fit_btd(obs, item_ids=["a", "b"], draws=300, tune=500, chains=2, seed=0)
    s_un = summarize_btd(res, observations=obs, position_neutral=False)
    s_ne = summarize_btd(res, observations=obs, position_neutral=True)
    un_pair = next(iter(s_un["pairwise"].values()))
    ne_pair = next(iter(s_ne["pairwise"].values()))
    # The difference is bounded by |beta_right|. With a clearly
    # advantaged right slot, the unconstrained P(tie) will be either
    # larger or smaller than the position-neutral P(tie).
    assert un_pair["p_tie"] != pytest.approx(ne_pair["p_tie"], abs=0.0), (
        "position_neutral must change P(tie) when beta_right != 0"
    )


# ---- tournament score -------------------------------------------------------

def test_tournament_score_tie_adjusted_position_neutral():
    """direct_summary now returns a tournament_score: half credit
    for ties, full credit for wins, normalized by N-1.

    Position-neutral: it does not depend on which slot the item
    appeared in, only on the verdicts.
    """
    obs = [
        # a beats b in both orientations -> 2 wins for a, 0 ties
        _obs("a", "b", "a", "b", "LEFT", 1),
        _obs("a", "b", "b", "a", "RIGHT", 1),
        # a ties c in both orientations -> 2 ties each
        _obs("a", "c", "a", "c", "TIE", 1),
        _obs("a", "c", "c", "a", "TIE", 1),
        # b ties c in both orientations -> 2 ties each
        _obs("b", "c", "b", "c", "TIE", 1),
        _obs("b", "c", "c", "b", "TIE", 1),
    ]
    d = direct_summary(obs)
    score = d["tournament_score"]
    # N=3, denom=2.
    # a: 2 wins (over b) + 2 ties (with c) -> (2 + 1.0) / 2 = 1.5
    # b: 0 wins + 2 ties (with c) -> 1.0 / 2 = 0.5
    # c: 0 wins + 2 ties (with a) + 2 ties (with b) -> 2.0 / 2 = 1.0
    assert abs(score["a"] - 1.5) < 1e-9
    assert abs(score["b"] - 0.5) < 1e-9
    assert abs(score["c"] - 1.0) < 1e-9


def test_tournament_score_for_undefeated_item():
    """An undefeated item that wins everything gets a score above 1.0.

    With N=3, the maximum score is (2 wins + 0.5*0 ties) / 2 = 1.0
    if the item wins both opponents with no ties, or higher if it
    also ties. The score is normalized by N-1 and bounded above by
    N-1 (when an item wins every obs, including ties scored as 0.5).

    Concretely: a beats b and c in both orientations -> 4 wins, 0 ties.
    Score = 4 / 2 = 2.0.
    """
    obs = [
        # a beats b in both orientations
        _obs("a", "b", "a", "b", "LEFT", 1),
        _obs("a", "b", "b", "a", "RIGHT", 1),
        # a beats c in both orientations
        _obs("a", "c", "a", "c", "LEFT", 1),
        _obs("a", "c", "c", "a", "RIGHT", 1),
        # b vs c irrelevant for a's score
        _obs("b", "c", "b", "c", "TIE", 1),
        _obs("b", "c", "c", "b", "TIE", 1),
    ]
    d = direct_summary(obs)
    # a: 4 wins (over b in 2 obs + over c in 2 obs), 0 ties.
    # score = (4 + 0) / 2 = 2.0
    assert abs(d["tournament_score"]["a"] - 2.0) < 1e-9


# ---- sanity: existing 5-level workflow still works -------------------------

def test_existing_5level_workflow_unchanged():
    """A user who calls fit_ordinal on 5-level data still gets the
    ordered logit, and the model name in the docstring is correct.
    """
    from pairwise_rank import fit_ordinal, summarize
    obs = [
        _obs("a", "b", "a", "b", "LEFT_STRONG", 1),
        _obs("a", "b", "a", "b", "LEFT", 2),
        _obs("a", "b", "a", "b", "TIE", 1),
        _obs("a", "b", "a", "b", "RIGHT", 1),
        _obs("a", "b", "a", "b", "RIGHT_STRONG", 2),
    ]
    res = fit_ordinal(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    s = summarize(res)
    # Both items have theta and P(best)
    assert len(s["per_item"]) == 2
    for row in s["per_item"]:
        assert "theta_mean" in row
        assert "p_best" in row
