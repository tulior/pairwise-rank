"""Protocol invariants: pair count, orientation, repeat, schedule, dedup/resume."""
from __future__ import annotations

import pytest

from pairwise_rank import (
    VERDICT_LEVELS,
    Observation,
    observation_key,
    make_schedule,
    run_tournament,
    save_observations_jsonl,
    load_observations_jsonl,
    # Imported lazily inside test bodies that need them, to avoid
    # pulling extra modules into every test:
)


# 1. pair count
def test_pair_count_is_n_choose_2():
    obs = make_schedule(["a", "b", "c", "d", "e"], repeats=1)
    # 5 items * 4 / 2 = 10 pairs * 2 orientations * 1 repeat = 20
    assert len(obs) == 20
    # All unique (a, b) pairs
    pairs = {(o.a, o.b) for o in obs}
    assert len(pairs) == 10


# 2. orientation count
def test_each_pair_has_two_orientations():
    obs = make_schedule(["a", "b", "c"], repeats=1)
    # 3 pairs * 2 orientations * 1 repeat = 6
    assert len(obs) == 6
    pair_oris = {(o.a, o.b, o.left) for o in obs}
    assert len(pair_oris) == 6
    # For each (a, b), both (a, b, a) and (a, b, b) appear
    for a, b in [("a", "b"), ("a", "c"), ("b", "c")]:
        assert (a, b, a) in pair_oris
        assert (a, b, b) in pair_oris


# 3. repeat count
def test_k_repeats_per_cell():
    obs = make_schedule(["a", "b", "c", "d"], repeats=3)
    # 6 pairs * 2 orientations * 3 repeats = 36
    assert len(obs) == 36
    repeats = {(o.a, o.b, o.left, o.right, o.repeat) for o in obs}
    assert len(repeats) == 36
    for o in obs:
        assert 1 <= o.repeat <= 3


# 4. deterministic schedule
def test_schedule_is_deterministic():
    obs1 = make_schedule(["alpha", "beta", "gamma"], repeats=2)
    obs2 = make_schedule(["alpha", "beta", "gamma"], repeats=2)
    assert obs1 == obs2


# 5. dedup / resume
def test_run_tournament_skips_existing_keys():
    """Passing existing observations back in skips those keys."""
    # Full schedule for 2 items, 2 repeats: 4 cells
    #   (a, b, a, b, 1) (a, b, a, b, 2) (a, b, b, a, 1) (a, b, b, a, 2)
    # Existing: one of them
    existing = [Observation(a="a", b="b", left="a", right="b", repeat=1, verdict="RIGHT")]
    called_keys = []

    def judge(left, right):
        return "LEFT"

    out = run_tournament(["a", "b"], judge, repeats=2, existing=existing)
    # 1 existing + 3 new = 4 total
    assert len(out) == 4
    # The existing observation is preserved
    existing_in_out = [o for o in out if observation_key(o) == ("a", "b", "a", "b", 1)]
    assert len(existing_in_out) == 1
    assert existing_in_out[0].verdict == "RIGHT"
    # The other 3 observations were completed by the judge
    new_ones = [o for o in out if o not in existing]
    assert len(new_ones) == 3
    assert all(o.verdict == "LEFT" for o in new_ones)


def test_run_tournament_rejects_invalid_verdict():
    def bad_judge(left, right):
        return "MAYBE"  # not in VERDICT_LEVELS
    with pytest.raises(ValueError):
        run_tournament(["a", "b"], bad_judge, repeats=1)


def test_observation_key_unique_per_cell():
    obs = make_schedule(["a", "b", "c"], repeats=2)
    keys = [observation_key(o) for o in obs]
    assert len(set(keys)) == len(obs)


def test_save_load_roundtrip(tmp_path):
    from pathlib import Path
    p = tmp_path / "obs.jsonl"
    obs = make_schedule(["a", "b", "c"], repeats=2)
    for o in obs:
        o.verdict = "TIE"
    save_observations_jsonl(p, obs)
    loaded = load_observations_jsonl(p)
    assert len(loaded) == len(obs)
    assert loaded[0].a == obs[0].a
    assert loaded[0].b == obs[0].b
    assert loaded[0].left == obs[0].left
    assert loaded[0].right == obs[0].right
    assert loaded[0].repeat == obs[0].repeat
    assert loaded[0].verdict == "TIE"


# ----------------------------------------------------------------------------
# Reasoning audit metadata
# ----------------------------------------------------------------------------

def test_judge_can_return_verdict_and_reasoning():
    """A judge_fn that returns (verdict, reasoning) stores both."""
    def judge(left, right):
        return ("LEFT", f"because {left} feels more native than {right}")

    obs = run_tournament(["a", "b", "c"], judge, repeats=2)
    assert len(obs) == 12  # 3 pairs * 2 orientations * 2 repeats
    assert all(o.reasoning for o in obs), "every observation should have non-empty reasoning"
    assert all(o.verdict == "LEFT" for o in obs)
    # Reasoning text refers to the actual pair.
    sample = obs[0]
    assert sample.left in sample.reasoning
    assert sample.right in sample.reasoning


def test_judge_can_still_return_just_verdict():
    """The simple str return shape is unchanged."""
    def judge(left, right):
        return "RIGHT"

    obs = run_tournament(["a", "b"], judge, repeats=2)
    assert all(o.verdict == "RIGHT" for o in obs)
    assert all(o.reasoning == "" for o in obs), "missing reasoning defaults to empty string"


def test_judge_returning_three_tuple_raises():
    """Only str or (str, str) are accepted from judge_fn."""
    def judge(left, right):
        return ("LEFT", "reasoning", "extra")
    with pytest.raises(ValueError):
        run_tournament(["a", "b"], judge, repeats=1)


def test_reasoning_survives_save_load(tmp_path):
    """Reasoning round-trips through JSONL."""
    obs = [Observation(
        a="a", b="b", left="a", right="b", repeat=1,
        verdict="TIE", reasoning="a multi-line\nreasoning trace\nwith details",
    )]
    p = tmp_path / "obs.jsonl"
    save_observations_jsonl(p, obs)
    loaded = load_observations_jsonl(p)
    assert loaded[0].reasoning == obs[0].reasoning


def test_load_backfills_missing_reasoning_with_default(tmp_path):
    """Rows written before the reasoning field was added still load."""
    p = tmp_path / "obs.jsonl"
    # Hand-write a row that omits the reasoning key, simulating old data.
    p.write_text(
        '{"a": "a", "b": "b", "left": "a", "right": "b", "repeat": 1, "verdict": "TIE"}\n'
    )
    loaded = load_observations_jsonl(p)
    assert len(loaded) == 1
    assert loaded[0].reasoning == ""


def test_load_ignores_unknown_keys(tmp_path):
    """Forward compatibility: a row with an extra unknown key still loads."""
    p = tmp_path / "obs.jsonl"
    p.write_text(
        '{"a": "a", "b": "b", "left": "a", "right": "b", "repeat": 1, '
        '"verdict": "TIE", "reasoning": "ok", "future_field": 42}\n'
    )
    loaded = load_observations_jsonl(p)
    assert len(loaded) == 1
    assert loaded[0].reasoning == "ok"


def test_reasoning_does_not_affect_dedup():
    """Two obs with the same key but different reasoning collapse to one;
    the existing row's reasoning is preserved verbatim."""
    existing = [Observation(
        a="a", b="b", left="a", right="b", repeat=1,
        verdict="TIE", reasoning="first-pass reasoning",
    )]

    def judge(left, right):
        return ("LEFT", "second-pass reasoning")

    out = run_tournament(["a", "b"], judge, repeats=1, existing=existing)
    matches = [o for o in out if observation_key(o) == ("a", "b", "a", "b", 1)]
    assert len(matches) == 1, "dedup must collapse on (a,b,left,right,repeat)"
    assert matches[0].reasoning == "first-pass reasoning", (
        "existing reasoning is preserved, not overwritten"
    )


def test_reasoning_does_not_affect_dedup_key():
    """The observation_key function never reads reasoning."""
    o1 = Observation(a="a", b="b", left="a", right="b", repeat=1, verdict="TIE", reasoning="x")
    o2 = Observation(a="a", b="b", left="a", right="b", repeat=1, verdict="TIE", reasoning="y")
    assert observation_key(o1) == observation_key(o2)


def test_reasoning_does_not_affect_fit(tmp_path):
    """The ranking model never reads reasoning. Fitting observations with
    and without reasoning on the same verdicts must produce the same
    posterior draws (deterministic given the same seed)."""
    import numpy as np
    from pairwise_rank import fit_btd

    base = [Observation(
        a="a", b="b", left="a", right="b", repeat=r, verdict="RIGHT",
    ) for r in range(1, 7)]
    with_reasoning = [
        Observation(**{**o.__dict__, "reasoning": f"trace for repeat {o.repeat}"})
        for o in base
    ]

    r1 = fit_btd(base, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    r2 = fit_btd(with_reasoning, item_ids=["a", "b"], draws=200, tune=300, chains=2, seed=0)
    np.testing.assert_allclose(r1.theta_draws, r2.theta_draws, atol=1e-6)
    np.testing.assert_allclose(r1.beta_right_draws, r2.beta_right_draws, atol=1e-6)
    np.testing.assert_allclose(r1.sigma_theta_draws, r2.sigma_theta_draws, atol=1e-6)
    np.testing.assert_allclose(r1.eta_tie_draws, r2.eta_tie_draws, atol=1e-6)


# 18. Default verdict scale is 3-level
def test_default_verdict_levels_is_3_level():
    from pairwise_rank import VERDICT_LEVELS, VERDICT_LEVELS_5
    assert VERDICT_LEVELS == ("LEFT", "TIE", "RIGHT")
    assert VERDICT_LEVELS_5 == ("LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG")
    # 3-level is the default for run_tournament
    assert len(VERDICT_LEVELS) == 3


# 19. collapse_to_3_level
def test_collapse_to_3_level():
    from pairwise_rank import collapse_to_3_level
    assert collapse_to_3_level("LEFT_STRONG") == "LEFT"
    assert collapse_to_3_level("RIGHT_STRONG") == "RIGHT"
    assert collapse_to_3_level("LEFT") == "LEFT"
    assert collapse_to_3_level("TIE") == "TIE"
    assert collapse_to_3_level("RIGHT") == "RIGHT"


# 20. run_tournament rejects 5-level verdicts by default
def test_run_tournament_rejects_5level_by_default():
    """With the new 3-level default, a 5-level verdict should be rejected."""
    def five_level_judge(left, right):
        return "LEFT_STRONG"
    with pytest.raises(ValueError, match="invalid verdict"):
        run_tournament(["a", "b"], five_level_judge, repeats=1)


# 21. run_tournament accepts 5-level verdicts when explicitly enabled
def test_run_tournament_accepts_5level_when_enabled():
    from pairwise_rank import VERDICT_LEVELS_5
    def five_level_judge(left, right):
        return "LEFT_STRONG"
    obs = run_tournament(
        ["a", "b"], five_level_judge, repeats=1,
        verdict_levels=VERDICT_LEVELS_5,
    )
    # 2 items, 1 repeat, 2 orientations = 2 obs
    assert len(obs) == 2
    assert all(o.verdict == "LEFT_STRONG" for o in obs)


# 22. 5-level data on disk loads correctly with default loader
def test_5level_data_on_disk_loads_fine():
    """Backward compat: legacy 5-level data loads without migration."""
    from pairwise_rank import load_observations_jsonl, fit_btd
    import json
    from pathlib import Path
    import tempfile
    rows = [
        {"a": "a", "b": "b", "left": "a", "right": "b", "repeat": 1, "verdict": "LEFT_STRONG"},
        {"a": "a", "b": "b", "left": "b", "right": "a", "repeat": 1, "verdict": "TIE"},
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        tmppath = f.name
    obs = load_observations_jsonl(Path(tmppath))
    assert obs[0].verdict == "LEFT_STRONG"
    assert obs[1].verdict == "TIE"
    # fit_btd collapses STRONG automatically
    result = fit_btd(obs, item_ids=["a", "b"], draws=200, tune=300, chains=1, seed=0)
    assert result.n == 2


# 23. M0 / fit_ordinal was removed; BTD is the only inference path.
