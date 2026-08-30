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


# ---------------------------------------------------------------------------
# PPC contract: shape, count, verdict domain, agreement reducer, reproducibility
# ---------------------------------------------------------------------------

def _ppc_fixture_obs():
    """A small fixture with three items and a mix of cell sizes
    (1, 2, 3, 4 repeats) used for the PPC contract tests."""
    obs = []
    # a vs b: 3 repeats, 2 orientations
    for r in range(1, 4):
        obs.append(_obs("a", "b", "a", "b", 1, r))
        obs.append(_obs("a", "b", "b", "a", 3, r))
    # a vs c: 2 repeats, 2 orientations
    for r in range(1, 3):
        obs.append(_obs("a", "c", "a", "c", 2, r))
        obs.append(_obs("a", "c", "c", "a", 2, r))
    # b vs c: 4 repeats, 2 orientations
    for r in range(1, 5):
        obs.append(_obs("b", "c", "b", "c", 1, r))
        obs.append(_obs("b", "c", "c", "b", 3, r))
    return obs


def test_ppc_shape_and_count_contract():
    """n_ppc=7 produces exactly 7 replicated datasets, each with
    exactly one generated verdict per observed row. The
    posterior-predictive array is (chain_subset, draw_subset, n_obs)
    with chain_subset * draw_subset == n_ppc.
    """
    import pymc as pm
    import xarray as xr
    import arviz as az
    from pairwise_rank.model import _build_model

    obs = _ppc_fixture_obs()
    result = fit(obs, item_ids=["a", "b", "c"], draws=200, tune=300, chains=1, seed=0)

    n_ppc = 7
    idata = result.idata
    n_chains = int(idata.posterior.sizes["chain"])
    n_draws = int(idata.posterior.sizes["draw"])
    S_total = n_chains * n_draws
    rng = np.random.default_rng(0)
    flat = rng.choice(S_total, size=n_ppc, replace=False)
    chain_idx = flat // n_draws
    draw_idx = flat % n_draws

    posterior_vars = {}
    for v in idata.posterior.data_vars:
        arr = np.asarray(idata.posterior[v].values)
        subset = arr[chain_idx, draw_idx]
        dims = ("chain", "draw") + tuple(idata.posterior[v].dims[2:])
        posterior_vars[v] = (dims, subset[np.newaxis, ...])
    sub_idata = az.InferenceData(posterior=xr.Dataset(posterior_vars))

    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    n_items = len(result.item_ids)
    model = _build_model(obs, item_to_idx, n_items)
    with model:
        ppc = pm.sample_posterior_predictive(
            sub_idata, var_names=["y_obs"],
            random_seed=0, progressbar=False,
        )
    arr = ppc.posterior_predictive["y_obs"]
    # Exactly n_ppc replicated datasets (1 chain * n_ppc draws).
    assert arr.sizes["chain"] * arr.sizes["draw"] == n_ppc
    # One generated verdict per observed row.
    assert arr.sizes["y_obs_dim_0"] == len(obs)


def test_ppc_verdict_domain():
    """Every generated value is a valid ordinal category in [0, 4]
    (the 5-level verdict space)."""
    obs = _ppc_fixture_obs()
    result = fit(obs, item_ids=["a", "b", "c"], draws=200, tune=300, chains=1, seed=0)
    ppc = posterior_predictive_check(result, obs, n_ppc=20, seed=0)
    # ppc doesn't expose the raw verdicts, so we re-run the
    # internals here. The function is exercised end-to-end above.
    import pymc as pm
    import xarray as xr
    import arviz as az
    from pairwise_rank.model import _build_model

    idata = result.idata
    n_chains = int(idata.posterior.sizes["chain"])
    n_draws = int(idata.posterior.sizes["draw"])
    S_total = n_chains * n_draws
    rng = np.random.default_rng(0)
    flat = rng.choice(S_total, size=20, replace=False)
    chain_idx = flat // n_draws
    draw_idx = flat % n_draws
    posterior_vars = {}
    for v in idata.posterior.data_vars:
        arr = np.asarray(idata.posterior[v].values)
        subset = arr[chain_idx, draw_idx]
        dims = ("chain", "draw") + tuple(idata.posterior[v].dims[2:])
        posterior_vars[v] = (dims, subset[np.newaxis, ...])
    sub_idata = az.InferenceData(posterior=xr.Dataset(posterior_vars))
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    model = _build_model(obs, item_to_idx, len(result.item_ids))
    with model:
        pred = pm.sample_posterior_predictive(
            sub_idata, var_names=["y_obs"],
            random_seed=0, progressbar=False,
        )
    verdicts = np.asarray(pred.posterior_predictive["y_obs"]).reshape(-1, len(obs))
    # 5-level ordinal: codes 0..4 (LEFT_STRONG, LEFT, TIE, RIGHT, RIGHT_STRONG).
    assert verdicts.min() >= 0
    assert verdicts.max() <= 4
    # All values are integers in [0, 4].
    assert np.all((verdicts >= 0) & (verdicts <= 4) & (verdicts == verdicts.astype(int)))


def test_cell_agreement_all_unanimous_is_one():
    """If every cell is unanimous, the agreement fraction is 1.0."""
    from collections import defaultdict
    from pairwise_rank.model import _cell_agreement

    # 3 cells of 2 unanimous observations each.
    ys = np.array([0, 0, 2, 2, 1, 1])
    cells = [[0, 1], [2, 3], [4, 5]]
    assert _cell_agreement(ys, cells, n_cells=3) == 1.0


def test_cell_agreement_no_unanimous_is_low():
    """If no cell is unanimous (and no single-repeat cells), the
    agreement fraction is 0.0."""
    from pairwise_rank.model import _cell_agreement

    # 3 cells of 2 observations each, all disagree.
    ys = np.array([0, 2, 0, 1, 2, 3])
    cells = [[0, 1], [2, 3], [4, 5]]
    assert _cell_agreement(ys, cells, n_cells=3) == 0.0


def test_cell_agreement_mixed_is_exact_known_value():
    """Hand-coded fixture with a known exact answer.

    Fixture: 4 cells total.
      cell 0: 2 obs, all 0  -> agree
      cell 1: 2 obs, mixed  -> disagree
      cell 2: 1 obs (vacuously unanimous) -> agree
      cell 3: 2 obs, all 3  -> agree
    Expected: 3/4 = 0.75
    """
    from pairwise_rank.model import _cell_agreement

    ys = np.array([0, 0, 1, 2, 1, 3, 3])
    cells = [[0, 1], [2, 3], [4], [5, 6]]
    assert _cell_agreement(ys, cells, n_cells=4) == 0.75


def test_cell_agreement_no_cells_is_zero():
    """An empty cells list returns 0.0 (defensive)."""
    from pairwise_rank.model import _cell_agreement

    assert _cell_agreement(np.array([]), [], n_cells=0) == 0.0


def test_ppc_reproducible_same_seed_same_result():
    """Same explicit random_seed produces the same returned PPC
    result. We check the structured output (not the raw verdicts)
    so the contract is what users actually see.
    """
    obs = _ppc_fixture_obs()
    result = fit(obs, item_ids=["a", "b", "c"], draws=200, tune=300, chains=1, seed=0)
    ppc1 = posterior_predictive_check(result, obs, n_ppc=30, seed=42)
    ppc2 = posterior_predictive_check(result, obs, n_ppc=30, seed=42)
    assert ppc1["observed"] == ppc2["observed"]
    assert ppc1["ppc_mean"] == ppc2["ppc_mean"]
    assert ppc1["ppc_hdi_90"] == ppc2["ppc_hdi_90"]
    assert ppc1["p_ppc_ge_observed"] == ppc2["p_ppc_ge_observed"]
    assert ppc1["n_ppc_draws"] == ppc2["n_ppc_draws"]


def test_ppc_n_ppc_preserved():
    """n_ppc=7 produces exactly 7 replicated datasets, not more."""
    obs = _ppc_fixture_obs()
    result = fit(obs, item_ids=["a", "b", "c"], draws=200, tune=300, chains=1, seed=0)
    ppc = posterior_predictive_check(result, obs, n_ppc=7, seed=0)
    assert ppc["n_ppc_draws"] == 7
