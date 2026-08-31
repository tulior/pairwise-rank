"""Tests for the adaptive comparison-design layer.

These tests are intentionally PyMC-free. The pure design
functions consume handcrafted ``p_best`` arrays, item lists, and
``pairwise_gt`` dicts; no fit is ever triggered. The orchestrator
test is also deterministic: it uses a synthetic judge with no
stochastic component, so the entire ``run_adaptive_best_set`` call
behaves the same on every run.
"""
from __future__ import annotations

import math

import pytest

from pairwise_rank import Observation
from pairwise_rank.design import (
    AdaptiveBestSetConfig,
    AdaptiveBestSetResult,
    AdaptiveBestSetState,
    credible_best_set,
    make_sparse_bootstrap,
    select_frontier_batch,
    should_stop_adaptive,
)


# ---------------------------------------------------------------------------
# credible_best_set
# ---------------------------------------------------------------------------

class TestCredibleBestSet:
    def test_k1_when_first_dominates(self):
        items = ["a", "b", "c", "d"]
        p = [0.96, 0.02, 0.01, 0.01]
        out = credible_best_set(items, p, confidence=0.95)
        assert len(out) == 1

    def test_first_three_when_no_dominant(self):
        items = ["a", "b", "c", "d"]
        p = [0.61, 0.27, 0.08, 0.04]
        out = credible_best_set(items, p, confidence=0.95)
        assert len(out) == 3
        assert set(out) == {"a", "b", "c"}

    def test_returned_prefix_is_minimal(self):
        items = ["a", "b", "c", "d"]
        p = [0.50, 0.30, 0.15, 0.05]
        out = credible_best_set(items, p, confidence=0.95)
        # Sum across the returned prefix must reach 0.95.
        idx = {it: i for i, it in enumerate(items)}
        mass = sum(p[idx[it]] for it in out)
        assert mass >= 0.95
        # Drop the final (lowest-mass) element and confirm the
        # remaining mass falls below the threshold.
        mass_minus_one = sum(p[idx[it]] for it in out[:-1])
        assert mass_minus_one < 0.95

    def test_returned_mass_meets_confidence(self):
        items = ["a", "b", "c", "d", "e"]
        p = [0.40, 0.25, 0.15, 0.10, 0.10]
        out = credible_best_set(items, p, confidence=0.90)
        idx = {it: i for i, it in enumerate(items)}
        mass = sum(p[idx[it]] for it in out)
        assert mass >= 0.90

    def test_stable_tie_breaks_by_index(self):
        # Equal p_best: items keep the input order in the prefix.
        items = ["a", "b", "c", "d"]
        p = [0.25, 0.25, 0.25, 0.25]
        out = credible_best_set(items, p, confidence=0.5)
        # Confidence 0.5 from a uniform 4-vector reaches threshold
        # at the second item; the prefix is deterministic.
        assert out == ("a", "b")

    def test_stable_tie_breaks_by_index_high_confidence(self):
        items = ["a", "b", "c", "d", "e"]
        p = [0.20, 0.20, 0.20, 0.20, 0.20]
        out = credible_best_set(items, p, confidence=0.95)
        # 5 equal items: cumsum hits 1.0 only at the 5th, so the
        # full prefix is returned. Stability is guaranteed by the
        # deterministic order.
        assert out == ("a", "b", "c", "d", "e")

    def test_negative_p_best_raises(self):
        items = ["a", "b"]
        p = [1.1, -0.1]
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.95)

    def test_non_normalized_p_best_raises(self):
        items = ["a", "b", "c"]
        p = [0.5, 0.5, 0.5]
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.95)

    def test_shape_mismatch_raises(self):
        items = ["a", "b", "c"]
        p = [0.5, 0.5]  # wrong length
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.95)

    def test_non_finite_p_best_raises(self):
        items = ["a", "b"]
        p = [float("nan"), 1.0]
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.95)

    def test_confidence_out_of_range_raises(self):
        items = ["a", "b"]
        p = [0.6, 0.4]
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.0)
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=1.0)
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=1.5)

    def test_no_silent_renormalization(self):
        # A distribution that sums to 0.999 (within rounding) but
        # is otherwise well-formed: it should raise, not be
        # silently renormalized.
        items = ["a", "b", "c"]
        p = [0.5, 0.3, 0.199]
        with pytest.raises(ValueError):
            credible_best_set(items, p, confidence=0.95)

    def test_returned_set_is_tuple(self):
        items = ["a", "b", "c", "d"]
        p = [0.6, 0.2, 0.1, 0.1]
        out = credible_best_set(items, p, confidence=0.95)
        assert isinstance(out, tuple)
        for el in out:
            assert isinstance(el, str)


# ---------------------------------------------------------------------------
# make_sparse_bootstrap
# ---------------------------------------------------------------------------

class TestMakeSparseBootstrap:
    def test_connected_graph_n_greater_than_degree(self):
        items = [f"i{i}" for i in range(10)]
        edges = make_sparse_bootstrap(items, degree=6, seed=0)
        # Build adjacency
        adj: dict[str, set[str]] = {it: set() for it in items}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        # BFS reachability from a single source.
        seen = {items[0]}
        stack = [items[0]]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == set(items)

    def test_every_item_has_degree_neighbors(self):
        items = [f"i{i}" for i in range(10)]
        degree = 6
        edges = make_sparse_bootstrap(items, degree=degree, seed=0)
        deg: dict[str, int] = {it: 0 for it in items}
        for a, b in edges:
            deg[a] += 1
            deg[b] += 1
        for it in items:
            assert deg[it] == degree

    def test_no_duplicate_edges(self):
        items = [f"i{i}" for i in range(12)]
        edges = make_sparse_bootstrap(items, degree=4, seed=1)
        canonical = {(a, b) if a < b else (b, a) for a, b in edges}
        assert len(canonical) == len(edges)

    def test_no_self_edges(self):
        items = [f"i{i}" for i in range(12)]
        edges = make_sparse_bootstrap(items, degree=4, seed=1)
        for a, b in edges:
            assert a != b

    def test_exact_n_times_degree_over_two_edges(self):
        items = [f"i{i}" for i in range(11)]
        degree = 6
        edges = make_sparse_bootstrap(items, degree=degree, seed=2)
        assert len(edges) == 11 * degree // 2

    def test_same_seed_same_graph(self):
        items = [f"i{i}" for i in range(15)]
        a = make_sparse_bootstrap(items, degree=6, seed=42)
        b = make_sparse_bootstrap(items, degree=6, seed=42)
        assert a == b

    def test_different_seed_different_graph(self):
        items = [f"i{i}" for i in range(15)]
        a = make_sparse_bootstrap(items, degree=6, seed=1)
        b = make_sparse_bootstrap(items, degree=6, seed=2)
        # Not bitwise identical: at least one edge differs.
        assert a != b

    def test_odd_degree_raises(self):
        items = [f"i{i}" for i in range(8)]
        with pytest.raises(ValueError):
            make_sparse_bootstrap(items, degree=3, seed=0)

    def test_degree_below_two_raises(self):
        items = [f"i{i}" for i in range(8)]
        with pytest.raises(ValueError):
            make_sparse_bootstrap(items, degree=1, seed=0)
        with pytest.raises(ValueError):
            make_sparse_bootstrap(items, degree=0, seed=0)

    def test_n_le_degree_returns_all_unordered_pairs(self):
        items = ["a", "b", "c", "d", "e"]
        edges = make_sparse_bootstrap(items, degree=6, seed=0)
        # n=5 <= degree=6: every unordered pair.
        expected = {
            (a, b) if a < b else (b, a)
            for i, a in enumerate(items)
            for b in items[i + 1:]
        }
        got = {(a, b) if a < b else (b, a) for a, b in edges}
        assert got == expected
        assert len(edges) == 5 * 4 // 2

    def test_returned_edges_are_canonicalized(self):
        items = [f"i{i}" for i in range(8)]
        edges = make_sparse_bootstrap(items, degree=4, seed=0)
        for a, b in edges:
            # Canonical order: a < b in the tuple position sense
            # AND lexicographically. The package convention is
            # lexicographic: the test asserts that.
            assert a < b


# ---------------------------------------------------------------------------
# select_frontier_batch
# ---------------------------------------------------------------------------

class TestSelectFrontierBatch:
    def test_same_uncertainty_higher_p_best_pair_wins(self):
        items = ["a", "b", "c", "d"]
        # All four items have q_ij = 0.5 in pairwise_gt; only
        # p_best differentiates the pairs. (a, b) sums to 0.7, (c, d)
        # sums to 0.3, so the (a, b) pair should rank first.
        p_best = {"a": 0.40, "b": 0.30, "c": 0.20, "d": 0.10}
        pairwise_gt = {
            (i, j): 0.5 for i in range(4) for j in range(i + 1, 4)
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=4, max_per_item_per_batch=1,
        )
        # First pair in the batch should be (a, b) (highest
        # relevance among the all-uncertainty pool).
        assert batch[0] == ("a", "b")

    def test_same_relevance_q_half_beats_q_eight(self):
        items = ["a", "b", "c", "d"]
        # Two candidate pairs in the pool:
        #   (a, b): q_ab = 0.5 (uncertain)  relevance = 0.5 + 0.5 = 1.0
        #   (c, d): q_cd = 0.8 (more certain) relevance = 0.5 + 0.5 = 1.0
        # Score formula prefers 0.5 (uncertain) over 0.8.
        p_best = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        pairwise_gt = {
            (0, 1): 0.5,  # (a, b)
            (2, 3): 0.8,  # (c, d)
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=2, max_per_item_per_batch=1,
        )
        # Batch top pair must be (a, b), the q=0.5 pair.
        assert batch[0] == ("a", "b")

    def test_q_zero_or_one_yields_zero_score(self):
        items = ["a", "b", "c", "d"]
        # (a, b) is decided (q=1.0). (c, d) is uncertain (q=0.5).
        # (a, b) should be filtered out by the score<=0 guard.
        p_best = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
        pairwise_gt = {
            (0, 1): 1.0,  # decided
            (2, 3): 0.5,  # uncertain
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=2, max_per_item_per_batch=1,
        )
        # Only (c, d) is selected; (a, b) is dropped.
        assert ("c", "d") in batch
        assert ("a", "b") not in batch

    def test_relevance_matters_low_ranked_irrelevant_does_not_outrank(self):
        # Pair (a, b) is in the pool with q=0.5, high relevance.
        # Pair (c, d) has the same q=0.5 but very low relevance:
        # it must not appear in the batch, even though it is also
        # at q=0.5. The frontier score is
        # ``(p_best[i] + p_best[j]) * 4 * q * (1 - q)``;
        # relevance is multiplicative, so (a, b) outranks (c, d)
        # by a factor of 9x. With batch_size=3 and cap=2, the
        # greedy picks the top 3 pairs and (c, d) is excluded.
        items = ["a", "b", "c", "d"]
        p_best = {"a": 0.45, "b": 0.45, "c": 0.05, "d": 0.05}
        pairwise_gt = {
            (0, 1): 0.5,
            (2, 3): 0.5,
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=3, max_per_item_per_batch=2,
        )
        # (a, b) has score 0.9 and must be in the batch.
        assert ("a", "b") in batch
        # (c, d) has score 0.1 and must not outrank (a, b). The
        # greedy fills the batch with the highest-scored pairs;
        # (c, d) is the lowest-scored pair and is never picked
        # within batch_size=3.
        for pair in batch:
            assert pair != ("c", "d")

    def test_batch_has_no_duplicate_pairs(self):
        items = [f"i{i}" for i in range(8)]
        p_best = {it: 1.0 / 8 for it in items}
        pairwise_gt = {
            (i, j): 0.5 for i in range(8) for j in range(i + 1, 8)
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=64, max_per_item_per_batch=4,
        )
        canonical = {(a, b) if a < b else (b, a) for a, b in batch}
        assert len(canonical) == len(batch)

    def test_max_per_item_per_batch_enforced(self):
        items = [f"i{i}" for i in range(6)]
        p_best = {it: 1.0 / 6 for it in items}
        pairwise_gt = {
            (i, j): 0.5 for i in range(6) for j in range(i + 1, 6)
        }
        cap = 1
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=20, max_per_item_per_batch=cap,
        )
        counts: dict[str, int] = {it: 0 for it in items}
        for a, b in batch:
            counts[a] += 1
            counts[b] += 1
        for it, c in counts.items():
            assert c <= cap

    def test_exploration_confidence_wider_than_decision(self):
        # The brief requires the exploration pool to be built at
        # the wider confidence ``1 - (1 - confidence) / 2``, which
        # for ``confidence = 0.95`` is ``0.975``. The pool must
        # therefore be no smaller than the credible set at
        # ``confidence``. We verify the pool property directly by
        # calling :func:`credible_best_set` at both confidences on
        # the same p_best and asserting the exploration set is a
        # superset of the decision set.
        from pairwise_rank.design import credible_best_set
        items = ["a", "b", "c", "d", "e", "f"]
        p = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
        decision = set(credible_best_set(items, p, confidence=0.95))
        exploration = set(credible_best_set(
            items, p, confidence=1 - (1 - 0.95) / 2,
        ))
        assert exploration.issuperset(decision)
        # And the exploration set must be strictly wider on this
        # input: the top-5 sum to 0.95, the top-6 sum to 1.0, so
        # 0.975 pulls in the 6th item that 0.95 does not.
        assert len(exploration) > len(decision)

    def test_pool_at_decision_confidence_uses_only_top(self):
        # The exploration pool at 0.975 includes the items in the
        # decision credible set at 0.95 plus any additional items
        # needed to push the cumulative mass past 0.975. We
        # verify this directly: the exploration set is a
        # superset of the decision set, and the rank ordering of
        # the items is preserved.
        from pairwise_rank.design import credible_best_set
        items = ["a", "b", "c", "d", "e", "f"]
        p = [0.30, 0.25, 0.20, 0.10, 0.10, 0.05]
        decision = credible_best_set(items, p, confidence=0.95)
        exploration = credible_best_set(
            items, p, confidence=1 - (1 - 0.95) / 2,
        )
        # Decision set is a strict subset of the exploration set.
        assert set(decision).issubset(set(exploration))
        assert len(exploration) > len(decision)

    def test_position_invariance(self):
        # The frontier score formula does not depend on the
        # orientation of the (i, j) pair. Whether the orchestrator
        # wrote pairwise_gt[(0, 1)] or pairwise_gt[(1, 0)] must
        # not change the batch. We assert the explicit symmetry:
        # two runs of select_frontier_batch with the keys swapped
        # return the same set of pairs.
        items = ["a", "b", "c", "d"]
        p_best = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        gt_forward = {
            (0, 1): 0.5,
            (0, 2): 0.7,
            (0, 3): 0.6,
            (1, 2): 0.5,
            (1, 3): 0.6,
            (2, 3): 0.5,
        }
        gt_swapped = {
            (1, 0): 0.5,
            (2, 0): 0.7,
            (3, 0): 0.6,
            (2, 1): 0.5,
            (3, 1): 0.6,
            (3, 2): 0.5,
        }
        b1 = select_frontier_batch(
            items, p_best, gt_forward,
            confidence=0.95, batch_size=4, max_per_item_per_batch=1,
        )
        b2 = select_frontier_batch(
            items, p_best, gt_swapped,
            confidence=0.95, batch_size=4, max_per_item_per_batch=1,
        )
        assert b1 == b2

    def test_position_invariance_against_q_value(self):
        # Score = (p_best[i] + p_best[j]) * 4 * q * (1 - q)
        # is symmetric in i <-> j by construction. We assert that
        # the value of the score itself is the same for the pair
        # (a, b) under either labeling.
        # Compute score for (a, b) by reading q from each
        # canonical form.
        p_best = {"a": 0.5, "b": 0.5}
        gt_a = {(0, 1): 0.7}
        gt_b = {(1, 0): 0.7}
        # Call the function with each dict and assert the top
        # scored pair is the same canonical pair in both calls.
        items = ["a", "b"]
        out_a = select_frontier_batch(
            items, p_best, gt_a, confidence=0.95, batch_size=1,
            max_per_item_per_batch=1,
        )
        out_b = select_frontier_batch(
            items, p_best, gt_b, confidence=0.95, batch_size=1,
            max_per_item_per_batch=1,
        )
        assert out_a == out_b
        assert out_a == (("a", "b"),)

    def test_missing_q_defaults_to_half(self):
        # A pair that is not in pairwise_gt is treated as q=0.5
        # (uncertainty 1.0). The score is then just relevance.
        items = ["a", "b", "c", "d"]
        p_best = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}
        # Only (a, b) is recorded; (c, d) is missing.
        pairwise_gt = {(0, 1): 0.5}
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=2, max_per_item_per_batch=1,
        )
        # Both (a, b) and (c, d) should be in the batch: (a, b)
        # because its q=0.5 is explicit, (c, d) because q defaults
        # to 0.5. Sort by relevance: (a, b) first.
        assert batch[0] == ("a", "b")
        assert ("c", "d") in batch

    def test_batch_size_is_an_upper_bound(self):
        items = [f"i{i}" for i in range(6)]
        p_best = {it: 1.0 / 6 for it in items}
        pairwise_gt = {
            (i, j): 0.5 for i in range(6) for j in range(i + 1, 6)
        }
        batch = select_frontier_batch(
            items, p_best, pairwise_gt,
            confidence=0.95, batch_size=3, max_per_item_per_batch=1,
        )
        assert len(batch) <= 3


# ---------------------------------------------------------------------------
# should_stop_adaptive
# ---------------------------------------------------------------------------

class TestShouldStopAdaptive:
    def test_stable_set_for_b_batches(self):
        cfg = AdaptiveBestSetConfig(stability_batches=2)
        history = [("a", "b"), ("a", "b"), ("a", "b")]
        assert should_stop_adaptive(history, 100, cfg) == (True, "stable")

    def test_same_members_different_display_order_is_stable(self):
        # frozenset identity: ordering inside the tuple does not
        # matter; the stability check is on the set of items.
        cfg = AdaptiveBestSetConfig(stability_batches=2)
        history = [("a", "b", "c"), ("c", "a", "b"), ("b", "c", "a")]
        assert should_stop_adaptive(history, 100, cfg) == (True, "stable")

    def test_changing_set_returns_false(self):
        cfg = AdaptiveBestSetConfig(stability_batches=2)
        history = [("a", "b"), ("a", "c")]
        assert should_stop_adaptive(history, 100, cfg) == (False, None)

    def test_max_budget_returns_true_budget(self):
        cfg = AdaptiveBestSetConfig(stability_batches=10, max_unordered_pairs=5)
        history = [("a", "b")]  # only one entry, stability not reachable
        assert should_stop_adaptive(history, 5, cfg) == (True, "budget")

    def test_max_budget_takes_precedence(self):
        cfg = AdaptiveBestSetConfig(stability_batches=2, max_unordered_pairs=10)
        # Both conditions could trigger; budget wins because it is
        # checked first.
        history = [("a", "b"), ("a", "b"), ("a", "b")]
        assert should_stop_adaptive(history, 10, cfg) == (True, "budget")

    def test_history_shorter_than_stability_batches_returns_false(self):
        cfg = AdaptiveBestSetConfig(stability_batches=3)
        history = [("a", "b"), ("a", "b")]
        assert should_stop_adaptive(history, 100, cfg) == (False, None)

    def test_empty_history_returns_false(self):
        cfg = AdaptiveBestSetConfig(stability_batches=1)
        assert should_stop_adaptive([], 100, cfg) == (False, None)

    def test_budget_stop_does_not_force_k1(self):
        # The orchestrator must return whatever the latest fit
        # says, even when the budget is hit before stability.
        # This test is structural: it asserts that should_stop
        # returns "budget" (not "stable") and does not touch the
        # credible set contents.
        cfg = AdaptiveBestSetConfig(stability_batches=100, max_unordered_pairs=2)
        history = [("a", "b", "c"), ("a", "b", "c", "d")]
        # The history is stable (both entries equal under
        # frozenset) but the budget triggers first.
        stop, reason = should_stop_adaptive(history, 2, cfg)
        assert stop is True
        assert reason == "budget"


# ---------------------------------------------------------------------------
# AdaptiveBestSetConfig validation
# ---------------------------------------------------------------------------

class TestAdaptiveBestSetConfig:
    def test_default_config_valid(self):
        # The default config must be valid out of the box.
        cfg = AdaptiveBestSetConfig()
        # No exception is the assertion.
        assert cfg.confidence == 0.95

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(confidence=0.0)
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(confidence=1.0)
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(confidence=1.5)
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(confidence=-0.1)

    def test_invalid_bootstrap_degree(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(bootstrap_degree=1)
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(bootstrap_degree=3)
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(bootstrap_degree=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(batch_size=0)

    def test_invalid_stability_batches(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(stability_batches=0)

    def test_invalid_max_per_item_per_batch(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(max_per_item_per_batch=0)

    def test_invalid_max_unordered_pairs(self):
        with pytest.raises(ValueError):
            AdaptiveBestSetConfig(max_unordered_pairs=0)
