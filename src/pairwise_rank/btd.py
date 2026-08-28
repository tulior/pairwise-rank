"""Bradley-Terry-Davidson (BTD) pairwise model.

The BTD model collapses the five-level ordinal verdict into three
outcomes:

    verdict in {LEFT_STRONG, LEFT}   -> "left wins"  (code 0)
    verdict == TIE                    -> "tie"        (code 1)
    verdict in {RIGHT, RIGHT_STRONG}  -> "right wins" (code 2)

This is the natural cross-check for the five-level ordered-logistic M0.
The Davidson extension handles the tie outcome by adding a tie parameter
nu that scales a geometric-mean tie region. The Rao-Kupper parameterization
used here is:

    p_i = exp(theta_i + beta_right_indicator_for_right_slot)
    p_j = exp(theta_j)
    nu  = exp(eta_tie)              # tie weight, > 0
    d   = sqrt(p_i * p_j)           # geometric mean

    P(left wins)  = p_j / (p_i + nu * d + p_j)
    P(tie)        = nu * d / (p_i + nu * d + p_j)
    P(right wins) = p_i / (p_i + nu * d + p_j)

A pair (i, j) is identified with (left, right) by the observation's
left/right fields, not by the canonical (a, b) ordering. beta_right
retains the same sign convention as the M0 model: positive beta_right
means the right slot is advantaged.

Identifiability:
    theta ~ ZeroSumNormal, so sum(theta) = 0 at every draw.
    eta_tie has no sum constraint; it is a single tie-weight parameter.

The model is fit to the same observations as M0. Use BTD as a secondary
ranking to cross-check that the winner is robust to collapsing STRONG
into ordinary wins/losses. The M0 model is the primary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pairwise_rank.btd requires pymc, pytensor, and arviz. "
        "Install with: pip install pymc pytensor arviz"
    ) from e

from .protocol import Observation, VERDICT_TO_CODE


# Map the five-level verdict onto the three-level BTD outcome.
# LEFT_STRONG, LEFT   -> 0 (left wins)
# TIE                 -> 1 (tie)
# RIGHT, RIGHT_STRONG -> 2 (right wins)
def _btd_code(verdict: str) -> int:
    if verdict in ("LEFT_STRONG", "LEFT"):
        return 0
    if verdict == "TIE":
        return 1
    if verdict in ("RIGHT", "RIGHT_STRONG"):
        return 2
    raise ValueError(f"unknown verdict: {verdict!r}")


# Count how many STRONG verdicts the BTD model is collapsing.
def _strong_count(obs: list[Observation]) -> dict[str, int]:
    n_left_strong = sum(1 for o in obs if o.verdict == "LEFT_STRONG")
    n_right_strong = sum(1 for o in obs if o.verdict == "RIGHT_STRONG")
    return {
        "left_strong": n_left_strong,
        "right_strong": n_right_strong,
        "total_collapsed": n_left_strong + n_right_strong,
    }


@dataclass
class BTDFitResult:
    """Posterior draws and item ordering for a fitted BTD model."""

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
    def eta_tie_draws(self) -> np.ndarray:
        """log tie-weight parameter, shape (S,). nu = exp(eta_tie)."""
        return self.idata.posterior["eta_tie"].values.flatten()

    @property
    def nu_draws(self) -> np.ndarray:
        return np.exp(self.eta_tie_draws)


# ----------------------------------------------------------------------------
# Fit
# ----------------------------------------------------------------------------

def _build_btd_model(
    observations: list[Observation],
    item_to_idx: dict[str, int],
    n: int,
):
    rights = np.array([item_to_idx[o.right] for o in observations], dtype=int)
    lefts = np.array([item_to_idx[o.left] for o in observations], dtype=int)
    ys = np.array([_btd_code(o.verdict) for o in observations], dtype=int)

    with pm.Model() as model:
        sigma_theta = pm.HalfNormal("sigma_theta", 1.0)
        theta = pm.ZeroSumNormal("theta", sigma=sigma_theta, shape=n)
        beta_right = pm.Normal("beta_right", 0.0, 0.5)
        eta_tie = pm.Normal("eta_tie", 0.0, 1.0)

        # p_i = exp(theta_right + beta_right), p_j = exp(theta_left)
        # For the Rao-Kupper likelihood we want
        #   log_p_i = theta[rights] + beta_right
        #   log_p_j = theta[lefts]
        # which gives
        #   d = sqrt(p_i * p_j) = exp((log_p_i + log_p_j) / 2)
        log_pi = theta[rights] + beta_right
        log_pj = theta[lefts]
        log_d = 0.5 * (log_pi + log_pj)

        # Numerically stable log-likelihood for the categorical:
        #   P(left wins)  = pj / (pi + nu*d + pj)
        #   P(tie)        = nu*d / (pi + nu*d + pj)
        #   P(right wins) = pi / (pi + nu*d + pj)
        # The shared denominator in log space is
        #   log Z = logsumexp(log_pi, log_pj + eta_tie, log_d + eta_tie - log 2)
        # But we can build the per-class log-probs directly and let
        # pymc's Categorical take care of the softmax.
        a = log_pj                              # log P(left wins)  (up to -log Z)
        b = log_d + eta_tie                     # log P(tie)
        c = log_pi                              # log P(right wins)
        # Use a custom log-probability rather than pm.Categorical with
        # pt.softmax (which is not exposed on this pytensor build).
        # log P(class k) = logit_k - logsumexp(logit_0, logit_1, logit_2)
        stacked = pt.stack([a, b, c], axis=1)
        log_Z = pt.logsumexp(stacked, axis=1)
        log_probs = stacked - log_Z[:, None]
        pm.Potential("y_obs_logp", pt.sum(log_probs[np.arange(ys.shape[0]), ys]))
    return model


def fit_btd(
    observations,
    *,
    item_ids: list[str] | None = None,
    draws: int = 2000,
    tune: int = 2500,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 0,
) -> BTDFitResult:
    """Fit the Bradley-Terry-Davidson pairwise model.

    The five-level ordinal verdicts are collapsed to three outcomes
    (left wins / tie / right wins). STRONG verdicts become ordinary
    wins/losses. Position bias (beta_right) is included for parity
    with the M0 model.

    observations: iterable of Observation. Rows with empty verdicts
                  are dropped.
    item_ids: optional explicit ordering. If None, ids are inferred
              and sorted.
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

    model = _build_btd_model(obs, item_to_idx, n)
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            nuts_sampler="numpyro",
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )

    return BTDFitResult(
        idata=idata,
        n=n,
        item_ids=list(item_ids),
        config={
            "draws": draws, "tune": tune, "chains": chains,
            "target_accept": target_accept, "seed": seed,
            "n_observations": len(obs),
            "strong_collapsed": _strong_count(obs),
        },
    )


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def summarize_btd(
    result: BTDFitResult,
    observations=None,
    hdi_prob: float = 0.9,
) -> dict:
    """All posterior summaries in one call. JSON-serializable.

    Keys mirror the M0 summarize() output where comparable, plus
    `eta_tie` and `nu` for the tie-weight parameter. Pairwise keys
    also include `p_left_wins` and `p_right_wins` from the BTD likelihood
    directly (which already incorporate tie probability).
    """
    theta = result.theta_draws
    S, n = theta.shape

    ranks = np.argsort(-theta, axis=1)
    rank_pos = np.array([np.where(ranks == i)[1] for i in range(n)])
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

    # Pairwise from theta.
    P = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            P[i, j] = float((theta[:, i] > theta[:, j]).mean())
            P[j, i] = 1.0 - P[i, j]

    # Pairwise BTD likelihood probabilities (account for ties and
    # beta_right position effect). These are the per-pair P(left wins)
    # and P(right wins) marginals under the model, averaged over all
    # observations for that pair (both orientations).
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    pairwise_lh: dict = {}
    if observations is not None:
        beta = result.beta_right_draws
        nu = result.nu_draws
        for o in observations:
            i = item_to_idx[o.left]
            j = item_to_idx[o.right]
            log_pi = theta[:, j] + beta      # right slot
            log_pj = theta[:, i]             # left slot
            log_d = 0.5 * (log_pi + log_pj)
            a = log_pj
            b = log_d + np.log(nu)
            c = log_pi
            m = np.maximum(np.maximum(a, b), c)
            ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
            Z = ea + eb + ec
            p_left = float((ea / Z).mean())
            p_tie = float((eb / Z).mean())
            p_right = float((ec / Z).mean())
            key = (i, j) if i < j else (j, i)
            entry = pairwise_lh.setdefault(key, {
                "sum_left": 0.0, "sum_tie": 0.0, "sum_right": 0.0, "n": 0,
            })
            entry["sum_left"] += p_left
            entry["sum_tie"] += p_tie
            entry["sum_right"] += p_right
            entry["n"] += 1
        # Convert to means
        for key, e in pairwise_lh.items():
            n_obs_for_pair = e["n"]
            pairwise_lh[key] = {
                "p_left_wins_mean": e["sum_left"] / n_obs_for_pair,
                "p_tie_mean": e["sum_tie"] / n_obs_for_pair,
                "p_right_wins_mean": e["sum_right"] / n_obs_for_pair,
            }

    pairwise = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = theta[:, i] - theta[:, j]
            hdi = az.hdi(d, hdi_prob=hdi_prob)
            key = (i, j)
            entry = {
                "p_i_gt_j": float(P[i, j]),
                "delta_mean": float(d.mean()),
                "delta_hdi": [float(hdi[0]), float(hdi[1])],
            }
            if key in pairwise_lh:
                lh = pairwise_lh[key]
                entry["p_left_wins"] = lh["p_left_wins_mean"]
                entry["p_tie"] = lh["p_tie_mean"]
                entry["p_right_wins"] = lh["p_right_wins_mean"]
            pairwise[key] = entry

    beta = result.beta_right_draws
    beta_hdi = az.hdi(beta, hdi_prob=hdi_prob)
    sigma = result.sigma_theta_draws
    sigma_hdi = az.hdi(sigma, hdi_prob=hdi_prob)
    eta_tie = result.eta_tie_draws
    eta_tie_hdi = az.hdi(eta_tie, hdi_prob=hdi_prob)
    nu = result.nu_draws
    nu_hdi = az.hdi(nu, hdi_prob=hdi_prob)

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
        "tie_parameter": {
            "eta_tie_mean": float(eta_tie.mean()),
            "eta_tie_hdi": [float(eta_tie_hdi[0]), float(eta_tie_hdi[1])],
            "nu_mean": float(nu.mean()),
            "nu_hdi": [float(nu_hdi[0]), float(nu_hdi[1])],
        },
    }
    if observations is not None:
        out["verdict_distribution_btd"] = _btd_verdict_counts(observations)
    return out


def _btd_verdict_counts(observations) -> dict[str, int]:
    out = {"left_wins": 0, "tie": 0, "right_wins": 0}
    for o in observations:
        if not o.verdict:
            continue
        c = _btd_code(o.verdict)
        if c == 0:
            out["left_wins"] += 1
        elif c == 1:
            out["tie"] += 1
        else:
            out["right_wins"] += 1
    return out


# ----------------------------------------------------------------------------
# Direct-tally summary (no model)
# ----------------------------------------------------------------------------

def direct_summary(observations) -> dict:
    """Per-item direct W/L/T and direct pairwise tallies.

    This is the third view in the standard report. It uses no model,
    just the raw observations. It depends on the dedup invariant:
    every (a, b) pair appears in both orientations.
    """
    from collections import defaultdict

    item_wins: dict[str, int] = defaultdict(int)
    item_losses: dict[str, int] = defaultdict(int)
    item_ties: dict[str, int] = defaultdict(int)
    pairs: dict[tuple[str, str], dict[str, int]] = {}

    for o in observations:
        if not o.verdict:
            continue
        c = _btd_code(o.verdict)
        # pair key (sorted, for direct pair table)
        p = tuple(sorted([o.a, o.b]))
        if p not in pairs:
            pairs[p] = {"wins_first": 0, "wins_second": 0, "ties": 0}
        if c == 1:
            item_ties[o.left] += 1
            item_ties[o.right] += 1
            pairs[p]["ties"] += 1
        elif c == 0:
            # left wins
            item_wins[o.left] += 1
            item_losses[o.right] += 1
            if o.left == p[0]:
                pairs[p]["wins_first"] += 1
            else:
                pairs[p]["wins_second"] += 1
        else:
            # right wins
            item_wins[o.right] += 1
            item_losses[o.left] += 1
            if o.right == p[0]:
                pairs[p]["wins_first"] += 1
            else:
                pairs[p]["wins_second"] += 1

    return {
        "per_item": {
            "wins": dict(item_wins),
            "losses": dict(item_losses),
            "ties": dict(item_ties),
        },
        "pairwise": {
            f"{p[0]},{p[1]}": v for p, v in pairs.items()
        },
        "n_observations": len([o for o in observations if o.verdict]),
        "n_left_strong": sum(1 for o in observations if o.verdict == "LEFT_STRONG"),
        "n_right_strong": sum(1 for o in observations if o.verdict == "RIGHT_STRONG"),
    }
