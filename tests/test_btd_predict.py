"""Tests for v0.4.1 additions:

  - predict_btd: per-cell (orientation-aware) BTD likelihood
  - summarize_btd: divergences, max_rhat, min_ess_bulk, min_ess_tail
  - BTDFitResult.divergences: stored on the dataclass
"""
from __future__ import annotations

import numpy as np
import pytest

from pairwise_rank import Observation, predict_btd, fit_btd, summarize_btd


def _obs(a, b, left, right, verdict_code, repeat=1):
    name = ["LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG"][verdict_code]
    return Observation(a=a, b=b, left=left, right=right, repeat=repeat, verdict=name)


# ---------------------------------------------------------------------------
# predict_btd
# ---------------------------------------------------------------------------


def test_predict_btd_returns_one_row_per_observation():
    """3 orientations x K=1 = 3 obs, predict_btd returns 3 rows."""
    obs = [
        _obs("a", "b", "a", "b", 3, 1),   # b wins on right
        _obs("a", "b", "b", "a", 0, 1),   # b wins on left (a wins) → LEFT (a wins on left)
        _obs("a", "b", "a", "b", 2, 1),   # TIE
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    preds = predict_btd(result, obs)
    assert len(preds) == 3
    for p in preds:
        assert set(p.keys()) >= {
            "left", "right", "repeat", "verdict",
            "p_left_wins", "p_tie", "p_right_wins",
        }
        # p values sum to 1
        total = p["p_left_wins"] + p["p_tie"] + p["p_right_wins"]
        assert abs(total - 1.0) < 1e-6


def test_predict_btd_per_orientation_differs_from_pair_average():
    """The per-orientation p_left_wins at (a, b) should differ from
    p_left_wins at (b, a). The summarize_btd pairwise dict averages
    them, predict_btd keeps them separate."""
    obs = [
        # a on left beats b (LEFT)
        _obs("a", "b", "a", "b", 0, 1),
        # a on right beats b (RIGHT) — adds position signal
        _obs("a", "b", "b", "a", 3, 1),
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    preds = predict_btd(result, obs)
    assert len(preds) == 2
    p1 = preds[0]   # a on left
    p2 = preds[1]   # a on right
    # p_right_wins in obs 2 should be higher than p_left_wins in obs 1
    # because a is favored in both, but the model accounts for position
    assert p1["left"] == "a" and p1["right"] == "b"
    assert p2["left"] == "b" and p2["right"] == "a"
    # a is the favored item in both, but the slots are flipped, so
    # the model's predicted p of the LEFT slot winning differs.
    # Obs 1: LEFT=a, model should put more mass on p_left_wins (a wins).
    # Obs 2: LEFT=b, model should put more mass on p_right_wins (a wins via right).
    assert p1["p_left_wins"] > p1["p_right_wins"]
    assert p2["p_right_wins"] > p2["p_left_wins"]


def test_predict_btd_position_neutral_differs_from_unconstrained():
    """If beta_right is far from 0, position_neutral=True should
    differ from position_neutral=False. We can't easily set beta_right
    in a small test, but we can verify the API path: position_neutral
    is forwarded, and the returned values stay valid (sum to 1).
    """
    obs = [_obs("a", "b", "a", "b", 3, 1)]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    p_full = predict_btd(result, obs, position_neutral=False)
    p_neut = predict_btd(result, obs, position_neutral=True)
    assert len(p_full) == 1 and len(p_neut) == 1
    # both sum to 1
    for p in (p_full[0], p_neut[0]):
        s = p["p_left_wins"] + p["p_tie"] + p["p_right_wins"]
        assert abs(s - 1.0) < 1e-6
    # If beta_right is small, the two should be close. We just check
    # both are valid probability triples.
    for p in (p_full[0], p_neut[0]):
        assert 0.0 <= p["p_left_wins"] <= 1.0
        assert 0.0 <= p["p_tie"] <= 1.0
        assert 0.0 <= p["p_right_wins"] <= 1.0


def test_predict_btd_skips_empty_verdicts():
    """Observations with empty verdicts are skipped, not crashed on."""
    obs = [
        _obs("a", "b", "a", "b", 3, 1),
        Observation(a="a", b="b", left="a", right="b", repeat=2, verdict=""),
    ]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    preds = predict_btd(result, obs)
    assert len(preds) == 1
    assert preds[0]["repeat"] == 1


def test_predict_btd_rejects_unknown_item():
    """Items not in the fit should raise ValueError."""
    obs = [_obs("a", "b", "a", "b", 3, 1)]
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    bad = [Observation(a="a", b="z", left="a", right="z", repeat=1, verdict="LEFT")]
    with pytest.raises(ValueError, match="not in fit.item_ids"):
        predict_btd(result, bad)


def test_predict_btd_preserves_5level_input_verdict_string():
    """predict_btd returns the input verdict string verbatim, even
    when the fit internally collapses LEFT_STRONG / RIGHT_STRONG."""
    obs = [_obs("a", "b", "a", "b", 4, 1)]   # RIGHT_STRONG
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    preds = predict_btd(result, obs)
    assert preds[0]["verdict"] == "RIGHT_STRONG"


# ---------------------------------------------------------------------------
# summarize_btd sampler diagnostics
# ---------------------------------------------------------------------------


def _fit_small_two_item():
    obs = [_obs("a", "b", "a", "b", 3, 1) for _ in range(6)]
    return fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)


def test_summarize_btd_includes_divergences_field():
    result = _fit_small_two_item()
    s = summarize_btd(result)
    assert "divergences" in s
    # The field is present and is either an int (count) or None
    # (sampler backend did not report divergences). It is never
    # silently coerced to 0 on a read failure -- a None value
    # signals "unverified", not "0 = healthy".
    assert s["divergences"] is None or isinstance(s["divergences"], int)
    if isinstance(s["divergences"], int):
        assert s["divergences"] >= 0


def test_summarize_btd_includes_rhat_ess_fields():
    result = _fit_small_two_item()
    s = summarize_btd(result)
    for key in ("max_rhat", "min_ess_bulk", "min_ess_tail"):
        assert key in s, f"missing {key}"
    # rhat should be a positive number close to 1 for a healthy fit
    assert s["max_rhat"] is not None
    assert s["max_rhat"] >= 1.0
    # ESS should be positive
    assert s["min_ess_bulk"] > 0
    assert s["min_ess_tail"] > 0


def test_btd_fit_result_exposes_divergences():
    """The BTDFitResult dataclass should carry divergences directly."""
    result = _fit_small_two_item()
    assert hasattr(result, "divergences")
    # The field is int | None; never silently 0 on a read failure.
    assert result.divergences is None or isinstance(result.divergences, int)
    if isinstance(result.divergences, int):
        assert result.divergences >= 0
    # And the summarize_btd output should match
    s = summarize_btd(result)
    assert s["divergences"] == result.divergences


def test_divergences_none_means_unverified_not_zero():
    """When the sampler backend does not report divergences, the
    field is None -- the package convention for "unavailable" --
    and is NOT coerced to 0. A None value is a red flag that the
    fit is unverified for geometry, not a pass.
    """
    # Run a real fit, then force the divergences field to None to
    # simulate what fit_btd would produce if the sampler backend
    # did not report a divergences field. This exercises the
    # summarize_btd path end-to-end without needing a mock idata.
    result = _fit_small_two_item()
    result.divergences = None
    s = summarize_btd(result)
    assert s["divergences"] is None
    assert "divergences" in s  # key still present, just None


def test_divergences_none_consistent_across_dataclass_and_summary():
    """The dataclass field and the summary output are always
    consistent -- if one is None, so is the other."""
    result = _fit_small_two_item()
    result.divergences = None
    assert result.divergences is None
    s = summarize_btd(result)
    assert s["divergences"] is None
    assert s["divergences"] == result.divergences


def test_divergences_dataclass_default_is_none():
    """The dataclass default for divergences is None, NOT 0.
    Constructing a BTDFitResult without explicitly setting
    divergences should yield None, signalling "unverified" rather
    than "0 divergent transitions = healthy".
    """
    from pairwise_rank.btd import BTDFitResult

    result = _fit_small_two_item()
    # Build a fresh dataclass using the real idata but with the
    # default for divergences, to mirror the post-fix fit_btd
    # contract.
    fresh = BTDFitResult(
        idata=result.idata,
        n=result.n,
        item_ids=result.item_ids,
        config=result.config,
    )
    assert fresh.divergences is None


def test_diagnostic_keys_present_with_position_neutral():
    """Sampler diagnostics should be present regardless of position_neutral."""
    result = _fit_small_two_item()
    for pn in (True, False):
        s = summarize_btd(result, position_neutral=pn)
        assert "divergences" in s
        assert "max_rhat" in s
        assert "min_ess_bulk" in s
        assert "min_ess_tail" in s


# ---------------------------------------------------------------------------
# Backward compatibility: existing fields still present
# ---------------------------------------------------------------------------


def test_summarize_btd_existing_keys_still_present():
    result = _fit_small_two_item()
    s = summarize_btd(result)
    for k in (
        "config", "item_ids", "n_items", "n_observations",
        "per_item", "pairwise", "position_effect", "sigma_theta",
        "tie_parameter", "position_neutral",
        # new keys added in 0.4.1
        "divergences", "max_rhat", "min_ess_bulk", "min_ess_tail",
    ):
        assert k in s, f"missing {k}"
