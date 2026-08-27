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
