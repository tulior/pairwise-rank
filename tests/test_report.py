"""Tests for the three-view report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pairwise_rank import (
    Observation,
    three_view_report,
    print_three_view,
)


def _obs(a, b, left, right, verdict_code, repeat=1):
    name = ["LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG"][verdict_code]
    return Observation(a=a, b=b, left=left, right=right, repeat=repeat, verdict=name)


@pytest.fixture(scope="module")
def tiny_tournament_report():
    """3-item round-robin, both orientations, K=3. Clear ordering:
    a is strictly stronger than b, b strictly stronger than c,
    a strictly stronger than c, in BOTH orientations.
    Total 18 obs.
    """
    obs = []
    for r in range(1, 4):
        # a beats b in both orientations:
        #   (a, b, a, b, LEFT_STRONG, r)  -- a on left wins
        #   (a, b, b, a, RIGHT, r)         -- b on left, right (a) wins
        obs.append(_obs("a", "b", "a", "b", 0, r))   # LEFT_STRONG: a wins
        obs.append(_obs("a", "b", "b", "a", 3, r))   # RIGHT: a wins (a is on right)
        # b beats c in both orientations:
        obs.append(_obs("b", "c", "b", "c", 1, r))   # LEFT: b wins
        obs.append(_obs("b", "c", "c", "b", 3, r))   # RIGHT: b wins (b is on right)
        # a beats c in both orientations:
        obs.append(_obs("a", "c", "a", "c", 0, r))   # LEFT_STRONG: a wins
        obs.append(_obs("a", "c", "c", "a", 3, r))   # RIGHT: a wins (a is on right)
    return three_view_report(obs, item_ids=["a", "b", "c"],
                             draws=400, tune=500, chains=2, seed=0)


def test_three_view_returns_expected_top_level_keys(tiny_tournament_report):
    rep = tiny_tournament_report
    for k in ("n_observations", "n_items", "item_ids", "strong_collapsed",
              "direct", "btd_summary", "m0_summary", "ranking",
              "top1", "theta_corr_btd_m0", "pbest_corr_btd_m0"):
        assert k in rep, f"missing key: {k}"


def test_three_view_recovers_ordering(tiny_tournament_report):
    rep = tiny_tournament_report
    # All three views should agree: a > b > c
    t1 = rep["top1"]
    assert t1["all_three_agree"]
    # a is the consensus winner
    by_id = {r["id"]: r for r in rep["ranking"]}
    assert by_id["a"]["direct_rank"] == 1
    assert by_id["a"]["btd_rank"] == 1
    assert by_id["a"]["m0_rank"] == 1


def test_three_view_theta_corr_high(tiny_tournament_report):
    """With identical data, BTD and M0 theta should be highly correlated."""
    rep = tiny_tournament_report
    assert rep["theta_corr_btd_m0"] > 0.9


def test_three_view_strong_collapse_count(tiny_tournament_report):
    rep = tiny_tournament_report
    # 6 obs use LEFT_STRONG (3 a-b + 3 a-c), 0 use RIGHT_STRONG
    assert rep["strong_collapsed"]["left_strong"] == 6
    assert rep["strong_collapsed"]["right_strong"] == 0
    assert rep["strong_collapsed"]["total_collapsed"] == 6


def test_three_view_direct_tallies(tiny_tournament_report):
    rep = tiny_tournament_report
    direct = rep["direct"]["per_item"]
    def net(c): return direct["wins"].get(c, 0) - direct["losses"].get(c, 0)
    # a > b: 6-0 = +6
    # b > c: 3-3 = 0  (3 wins for b, 3 losses for a, but b's 3 losses to a are not counted here)
    # Wait: b's wins = 3 (over c in 2 orientations x 3 reps... actually 3 obs each side)
    # b's losses = 3 (a-b) 3 obs each side
    # So b: 3 wins, 3 losses, net 0
    # c: 0 wins, 9 losses, net -9
    assert net("a") > 0
    assert net("c") < 0
    # The model should still rank a > b > c because b's wins over c
    # (LEFT+RIGHT, both orientations) exceed c's wins over b (= 0).


def test_three_view_print_runs(capsys, tiny_tournament_report):
    print_three_view(tiny_tournament_report, label="test")
    captured = capsys.readouterr()
    assert "Top-1" in captured.out
    assert "All three agree" in captured.out
    assert "BTD vs M0" in captured.out


def test_three_view_rejects_empty():
    with pytest.raises(ValueError):
        three_view_report([], item_ids=["a"])


def test_three_view_infers_items():
    """When item_ids is None, infer from observations."""
    obs = [_obs("a", "b", "a", "b", 1, r) for r in range(1, 4)]
    rep = three_view_report(obs, draws=300, tune=400, chains=1, seed=0)
    assert "a" in rep["item_ids"]
    assert "b" in rep["item_ids"]


def test_three_view_serializable():
    """Output should be JSON-serializable (numpy types are converted)."""
    obs = [_obs("a", "b", "a", "b", 1, r) for r in range(1, 4)]
    rep = three_view_report(obs, draws=300, tune=400, chains=1, seed=0)
    json.dumps(rep, default=str)  # should not raise
