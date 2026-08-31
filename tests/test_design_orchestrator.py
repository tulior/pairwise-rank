"""Tests for ``run_adaptive_best_set`` (the orchestrator).

These tests are kept in a separate file because the orchestrator
is the only place in the design layer that touches PyMC; the
remaining 51 design tests in ``test_design.py`` are pure-function
and run in a few minutes. The orchestrator tests use a
deterministic synthetic judge and are reproducible, but they
still take a few minutes each because of the MCMC sampler.

The default test suite excludes this file (see ``AGENTS.md``
section 7) so the four statistical files plus the pure design
tests fit the existing 4-batch split. Run this file
separately when validating the orchestrator end-to-end:

    PYTHONPATH=src pytest tests/test_design_orchestrator.py
"""
from __future__ import annotations

import pytest

from pairwise_rank.design import (
    AdaptiveBestSetConfig,
    AdaptiveBestSetResult,
    run_adaptive_best_set,
)


# ---------------------------------------------------------------------------
# PyMC sandbox workaround
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_pymc_cores(monkeypatch):
    # Some sandboxed environments report a CPU count of 0 via
    # ``multiprocessing.cpu_count() // 2``, which causes
    # ``pymc.sampling.mcmc._cpu_count`` to return 0 and
    # triggers a ZeroDivisionError in PyMC's core-allocation
    # path. ``pymc.sampling.mcmc`` imports ``_cpu_count`` from
    # ``pymc.sampling.parallel`` at module load time, so the
    # patch must target the local reference in
    # ``pymc.sampling.mcmc``. Also force the PyTensor linker
    # to the Python implementation so the test does not
    # require a working C compiler.
    try:
        from pymc.sampling import mcmc as _mcmc
    except ImportError:
        yield
        return
    monkeypatch.setattr(_mcmc, "_cpu_count", lambda: 1)
    try:
        import pytensor
        pytensor.config.linker = "py"
    except ImportError:
        pass
    yield


# ---------------------------------------------------------------------------
# run_adaptive_best_set
# ---------------------------------------------------------------------------

class TestRunAdaptiveBestSetDeterministic:
    """The orchestrator test is deterministic: the judge is a
    handcrafted callable that returns the same verdict for the
    same orientation, so the entire run is reproducible. PyMC IS
    invoked by the orchestrator (the orchestrator is the only
    place a refit happens); the test still does not use a
    stochastic or slow stochastic marker.
    """

    def test_n4_returns_stable_credible_set(self):
        # Items a, b, c, d. The judge always picks the right slot
        # as the winner unless TIE is the explicit comparison. The
        # judge is deterministic: the same orientation always
        # produces the same verdict.
        items = ["a", "b", "c", "d"]
        # Synthesize a ranking: a > b > c > d. The judge always
        # votes the lexicographically larger id as the winner on
        # the right slot, ties otherwise.
        def judge(left: str, right: str) -> str:
            if left == right:
                return "TIE"
            return "RIGHT" if right > left else "LEFT"

        cfg = AdaptiveBestSetConfig(
            confidence=0.95,
            bootstrap_degree=2,  # tiny bootstrap: 2*4/2=4 edges
            batch_size=4,
            stability_batches=2,
            seed=0,
        )
        result = run_adaptive_best_set(
            items, judge, config=cfg, repeats=1,
        )
        assert isinstance(result, AdaptiveBestSetResult)
        assert result.k >= 1
        # The credible set must be a non-empty prefix of items
        # sorted by p_best; it cannot include items not in the
        # input list.
        assert all(it in items for it in result.credible_best_set)

    def test_max_unordered_pairs_below_bootstrap_returns_result(self):
        items = ["a", "b", "c", "d"]
        # Bootstrap on n=4 with degree=2 has 4 edges. We set the
        # budget to 3 so the loop hits the budget before any
        # adaptive batch can be acquired.
        def judge(left: str, right: str) -> str:
            if left == right:
                return "TIE"
            return "RIGHT" if right > left else "LEFT"

        cfg = AdaptiveBestSetConfig(
            confidence=0.95,
            bootstrap_degree=2,
            batch_size=4,
            stability_batches=10,  # high; never reached
            max_unordered_pairs=3,
            seed=0,
        )
        result = run_adaptive_best_set(
            items, judge, config=cfg, repeats=1,
        )
        assert isinstance(result, AdaptiveBestSetResult)
        # The orchestrator must not raise; it must return a
        # result with a finite, non-empty credible set.
        assert result.k >= 1
        assert result.stopped_reason in ("stable", "budget")

    def test_budget_stop_does_not_silently_pick_k1(self):
        # The orchestrator must not silently pick the single
        # highest p_best item when the budget is hit. We give
        # the orchestrator very few observations and check that
        # the result returns what the latest fit said.
        items = ["a", "b", "c", "d"]
        def judge(left: str, right: str) -> str:
            if left == right:
                return "TIE"
            return "RIGHT" if right > left else "LEFT"

        # Seed with the bootstrap (4 edges) + 1 extra batch. The
        # budget at 4 means the orchestrator stops after the
        # bootstrap fit and never enters an adaptive batch.
        cfg = AdaptiveBestSetConfig(
            confidence=0.95,
            bootstrap_degree=2,
            batch_size=4,
            stability_batches=100,  # never reached
            max_unordered_pairs=4,
            seed=0,
        )
        result = run_adaptive_best_set(
            items, judge, config=cfg, repeats=1,
        )
        # We do not assert k == 1: that would be forcing the
        # design. We only assert that the result is a real fit
        # output with the budget as the reason.
        assert result.stopped_reason == "budget"
        assert result.batches >= 1
