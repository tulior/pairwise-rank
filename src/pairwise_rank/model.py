"""Bayesian ordered-logistic (five-level) pairwise model.

Status: OPTIONAL. This is the legacy 5-level ordered logit. The
recommended model for new code is `pairwise_rank.fit_btd` (the
Bradley-Terry-Davidson model in btd.py), which uses a 3-level
win/tie/loss verdict scale. BTD is simpler, has fewer parameters,
and matches observed behavior more closely on most real
tournaments; the ordered logit is preserved for cases where
preference intensity is genuinely elicited and the extra
categories are materially populated.

Use `fit_ordinal` only when at least one of the following is true:
  - STRONG verdicts occur often enough to be informative
    (a rough trigger: STRONG > 10-15% of non-ties);
  - STRONG vs ordinary wins show demonstrably different behavior
    that the 3-level model cannot capture;
  - the prompt deliberately elicits intensity;
  - BTD and direct evidence show unresolved structure that the
    ordinal information might explain;
  - you are specifically studying whether preference magnitude
    matters.

If you are using `fit_ordinal` routinely on data where STRONG
verdicts are < 2% of non-ties, switch to `fit_btd`. The two
models produce nearly identical rankings on such data
(r_theta > 0.99, r_P(best) > 0.99 in our experiments), and the
simpler model is preferred.

Model:
    eta = theta_right - theta_left + beta_right
    theta ~ ZeroSumNormal(sigma = sigma_theta)        # sum-to-zero
    sigma_theta ~ HalfNormal(1.0)
    beta_right ~ Normal(0, 0.5)                       # right-slot position effect
    cutpoints: 3 positive gaps via softplus, then zero-centered
    y_obs ~ OrderedLogistic(eta, cutpoints)          # 0..4

Sign conventions:
    - Larger theta means stronger in general.
    - beta_right > 0 means the right slot is advantaged.
    - verdict_code: 0 = LEFT_STRONG, 1 = LEFT, 2 = TIE, 3 = RIGHT, 4 = RIGHT_STRONG.
    - P(left wins) = P(y in {0,1}) = sigmoid(c_1 - eta) using c_1 (upper bound of LEFT region).
    - P(TIE) = P(y = 2) = sigmoid(c_2 - eta) - sigmoid(c_1 - eta).

Public surface: fit_ordinal, summarize, posterior_predictive_check.
The legacy name `fit` is preserved as an alias for fit_ordinal and
emits a DeprecationWarning; new code should use fit_ordinal
explicitly or fit_btd (the default).
File I/O is the caller's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pairwise_rank.model requires pymc, pytensor, and arviz. "
        "Install with: pip install pymc pytensor arviz"
    ) from e

from .protocol import Observation, observation_key, verdict_to_code


@dataclass
class FitResult:
    """Posterior draws and item ordering for a fitted model."""

    idata: object
    n: int
    item_ids: list[str]
    config: dict = field(default_factory=dict)

    @property
    def theta_draws(self) -> np.ndarray:
        """Posterior draws of theta, shape (S, n)."""
        return self.idata.posterior["theta"].values.reshape(-1, self.n)

    @property
    def beta_right_draws(self) -> np.ndarray:
        return self.idata.posterior["beta_right"].values.flatten()

    @property
    def sigma_theta_draws(self) -> np.ndarray:
        return self.idata.posterior["sigma_theta"].values.flatten()

    @property
    def cutpoint_draws(self) -> np.ndarray:
        """Posterior draws of cutpoints, shape (S, 4)."""
        return self.idata.posterior["cutpoints"].values.reshape(-1, 4)


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------

def _build_model(observations: list[Observation], item_to_idx: dict[str, int], n: int):
    rights = np.array([item_to_idx[o.right] for o in observations], dtype=int)
    lefts = np.array([item_to_idx[o.left] for o in observations], dtype=int)
    ys = np.array([verdict_to_code(o.verdict) for o in observations], dtype=int)

    with pm.Model() as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        theta = pm.ZeroSumNormal("theta", sigma=sigma_theta, shape=n)
        beta_right = pm.Normal("beta_right", 0.0, 0.5)

        # Ordered asymmetric cutpoints: 3 positive gaps via softplus, then centered.
        gap_raw = pm.Normal("cutpoint_gap_raw", 0.0, 0.7, shape=3)
        gaps = pm.Deterministic("cutpoint_gaps", pt.softplus(gap_raw))
        k0 = pt.zeros(1)
        k_rest = pt.cumsum(gaps)
        k_uncentered = pt.concatenate([k0, k_rest])
        cutpoints = pm.Deterministic("cutpoints", k_uncentered - pt.mean(k_uncentered))

        eta = theta[rights] - theta[lefts] + beta_right
        pm.OrderedLogistic("y_obs", eta=eta, cutpoints=cutpoints, observed=ys)
    return model


def fit_ordinal(
    observations: Iterable[Observation],
    *,
    item_ids: list[str] | None = None,
    draws: int = 2000,
    tune: int = 2500,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 0,
) -> FitResult:
    """Fit the five-level ordered-logistic pairwise model.

    Status: OPTIONAL / LEGACY. Use fit_btd() as the default probabilistic
    model for new code. See module docstring for when to use this
    model instead.

    observations: list of Observation. Rows with empty verdicts are dropped.
    item_ids: optional explicit ordering. If None, ids are inferred and sorted.
    """
    obs = [o for o in observations if o.verdict]
    if not obs:
        raise ValueError("no completed observations to fit")

    if item_ids is None:
        seen = []
        seen_set = set()
        for o in obs:
            for i in (o.a, o.b, o.left, o.right):
                if i not in seen_set:
                    seen.append(i)
                    seen_set.add(i)
        item_ids = sorted(seen)
    n = len(item_ids)
    item_to_idx = {i: k for k, i in enumerate(item_ids)}

    model = _build_model(obs, item_to_idx, n)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            nuts_sampler="numpyro",
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )

    return FitResult(
        idata=idata,
        n=n,
        item_ids=list(item_ids),
        config={
            "draws": draws, "tune": tune, "chains": chains,
            "target_accept": target_accept, "seed": seed,
            "n_observations": len(obs),
        },
    )


# Backward-compatible alias. The legacy name `fit` still works but
# emits a DeprecationWarning at runtime. New code should use
# fit_ordinal() or fit_btd().
import warnings as _warnings


def fit(
    observations: Iterable[Observation],
    *,
    item_ids: list[str] | None = None,
    draws: int = 2000,
    tune: int = 2500,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 0,
) -> FitResult:
    """DEPRECATED: alias for fit_ordinal(). Use fit_ordinal() or fit_btd().

    Kept for backward compatibility. New code should call fit_btd()
    (the BTD model in btd.py) as the default probabilistic model.
    """
    _warnings.warn(
        "pairwise_rank.fit is deprecated. Use fit_ordinal() for the "
        "5-level ordered-logistic model (now optional/legacy) or "
        "fit_btd() for the default 3-level BTD model. See module "
        "docstring of model.py and btd.py for guidance.",
        DeprecationWarning,
        stacklevel=2,
    )
    return fit_ordinal(
        observations,
        item_ids=item_ids,
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        seed=seed,
    )


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def _verdict_distribution(observations: list[Observation]) -> dict[str, int]:
    out = {v: 0 for v in ("LEFT_STRONG", "LEFT", "TIE", "RIGHT", "RIGHT_STRONG")}
    for o in observations:
        if o.verdict in out:
            out[o.verdict] += 1
    return out


def _pairwise_p(theta_draws: np.ndarray) -> np.ndarray:
    """P(theta_i > theta_j) for i < j. Antisymmetric across diagonal."""
    n = theta_draws.shape[1]
    out = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = float((theta_draws[:, i] > theta_draws[:, j]).mean())
            out[j, i] = 1.0 - out[i, j]
    return out


def summarize(
    result: FitResult,
    observations: list[Observation] | None = None,
    hdi_prob: float = 0.9,
) -> dict:
    """All posterior summaries in one call. JSON-serializable.

    Keys:
      config: FitResult.config
      item_ids: list[str]
      n_items, n_observations
      per_item: list of {id, theta_mean, theta_hdi, p_best, p_top2, expected_rank}
      pairwise: nested dict {(i,j): {p_i_gt_j, delta_mean, delta_hdi}}
      position_effect: {beta_right_mean, beta_right_hdi}
      sigma_theta: {mean, hdi}
      cutpoints: {mean (length 4), per_draw_zero_sum: bool}
      verdict_distribution: counts (only if observations provided)
    """
    theta = result.theta_draws
    S, n = theta.shape

    ranks = np.argsort(-theta, axis=1)
    rank_pos = np.array([np.where(ranks == i)[1] for i in range(n)])  # (n, S)
    p_best = np.array([(np.argmax(theta, axis=1) == i).mean() for i in range(n)])
    p_top2 = np.array([(rank_pos[i] <= 1).mean() for i in range(n)])
    mean_rank = rank_pos.mean(axis=1) + 1

    per_item = []
    for i in range(n):
        hdi = az.hdi(theta[:, i], hdi_prob=hdi_prob)
        per_item.append({
            "id": result.item_ids[i],
            "theta_mean": float(theta[:, i].mean()),
            "theta_hdi": [float(hdi[0]), float(hdi[1])],
            "p_best": float(p_best[i]),
            "p_top2": float(p_top2[i]),
            "expected_rank": float(mean_rank[i]),
        })

    P = _pairwise_p(theta)
    pairwise = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = theta[:, i] - theta[:, j]
            hdi = az.hdi(d, hdi_prob=hdi_prob)
            pairwise[(i, j)] = {
                "p_i_gt_j": float(P[i, j]),
                "delta_mean": float(d.mean()),
                "delta_hdi": [float(hdi[0]), float(hdi[1])],
            }

    beta = result.beta_right_draws
    beta_hdi = az.hdi(beta, hdi_prob=hdi_prob)
    sigma = result.sigma_theta_draws
    sigma_hdi = az.hdi(sigma, hdi_prob=hdi_prob)
    cps = result.cutpoint_draws
    cp_hdis = [az.hdi(cps[:, k], hdi_prob=hdi_prob).tolist() for k in range(4)]

    out = {
        "config": result.config,
        "item_ids": list(result.item_ids),
        "n_items": n,
        "n_observations": (len(observations) if observations is not None
                            else result.config.get("n_observations", 0)),
        "per_item": per_item,
        "pairwise": {f"{i},{j}": v for (i, j), v in pairwise.items()},
        "position_effect": {
            "beta_right_mean": float(beta.mean()),
            "beta_right_hdi": [float(beta_hdi[0]), float(beta_hdi[1])],
        },
        "sigma_theta": {
            "mean": float(sigma.mean()),
            "hdi": [float(sigma_hdi[0]), float(sigma_hdi[1])],
        },
        "cutpoints": {
            "mean": cps.mean(axis=0).tolist(),
            "hdi_each": cp_hdis,
        },
    }
    if observations is not None:
        out["verdict_distribution"] = _verdict_distribution(observations)
    return out


# ----------------------------------------------------------------------------
# Posterior predictive check
# ----------------------------------------------------------------------------

def posterior_predictive_check(
    result: FitResult,
    observations: list[Observation],
    *,
    n_ppc: int = 1000,
    statistic: str = "agreement",
    seed: int = 0,
) -> dict:
    """One-shot posterior predictive check on a single repeat-agreement statistic.

    statistic='agreement' computes, for each (a, b, orientation) cell, the
    fraction of posterior predictive draws in which all K repeats agree
    on the verdict. Returns observed value, PPC distribution, and tail
    probability.

    A single PPC statistic does not establish full calibration. Use as
    one diagnostic, not a certification.
    """
    from collections import defaultdict
    from scipy.special import expit

    rng = np.random.default_rng(seed)
    theta = result.theta_draws
    beta = result.beta_right_draws
    cutpoints = result.cutpoint_draws
    S = theta.shape[0]

    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    rights = np.array([item_to_idx[o.right] for o in observations], dtype=int)
    lefts = np.array([item_to_idx[o.left] for o in observations], dtype=int)
    ys = np.array([verdict_to_code(o.verdict) for o in observations], dtype=int)

    by_cell = defaultdict(list)
    for k, o in enumerate(observations):
        by_cell[(o.a, o.b, o.left, o.right)].append(k)
    cells = list(by_cell.values())
    n_cells = len(cells)

    ppc_stat = np.zeros(n_ppc)
    for ps in range(n_ppc):
        s = int(rng.integers(0, S))
        n_agree = 0
        for cell_idx in cells:
            if len(cell_idx) < 2:
                continue
            eta = theta[s, rights[cell_idx[0]]] - theta[s, lefts[cell_idx[0]]] + beta[s]
            cp = cutpoints[s]
            p0 = expit(cp[0] - eta)
            p1 = expit(cp[1] - eta) - p0
            p2 = expit(cp[2] - eta) - expit(cp[1] - eta)
            p3 = expit(cp[3] - eta) - expit(cp[2] - eta)
            p4 = 1.0 - expit(cp[3] - eta)
            probs = np.clip(np.array([p0, p1, p2, p3, p4]), 1e-10, 1.0)
            probs = probs / probs.sum()
            votes = rng.choice(5, size=len(cell_idx), p=probs)
            if (votes == votes[0]).all():
                n_agree += 1
        ppc_stat[ps] = n_agree / max(1, n_cells)

    n_agree_obs = sum(1 for cell_idx in cells
                      if len(set(ys[k] for k in cell_idx)) == 1)
    obs_stat = n_agree_obs / max(1, n_cells)

    hdi = az.hdi(ppc_stat, hdi_prob=0.9)
    p_tail = float((ppc_stat >= obs_stat).mean())

    return {
        "statistic": statistic,
        "observed": float(obs_stat),
        "ppc_mean": float(ppc_stat.mean()),
        "ppc_hdi_90": [float(hdi[0]), float(hdi[1])],
        "p_ppc_ge_observed": p_tail,
        "n_ppc_draws": n_ppc,
    }
