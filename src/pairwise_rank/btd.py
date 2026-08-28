"""Bradley-Terry-Davidson (BTD) pairwise model.

The BTD model is the default probabilistic ranking model. It uses a
3-outcome (LEFT / TIE / RIGHT) verdict scale and is fit jointly to all
observations with a sum-to-zero item-strength prior.

# Model family

This is the Davidson (1970) extension of Bradley-Terry, with a tie
term proportional to ``nu * sqrt(lambda_i * lambda_j)`` and ``nu > 0``:

    lambda_i = exp(theta[i] + beta_right_if_i_on_right)
    lambda_j = exp(theta[j])

    P(i beats j)        = lambda_i / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
    P(i ties j)         = nu * sqrt(lambda_i * lambda_j) / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
    P(i loses to j)     = lambda_j / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))

The tie term is a single global ``nu = exp(eta_tie)``. ``nu = 1`` is
the symmetric tie prior. The likelihood is symmetric in (i, j) and
reduces to Bradley-Terry as ``nu -> 0`` (forced-decision regime).

Note: this is **not** the Rao-Kupper parameterization. Rao-Kupper
uses a tie term of the form ``nu * (lambda_i + lambda_j) / 2``; Davidson
uses the geometric-mean form. The two are statistically distinguishable
on real data. We use the Davidson form here.

# Implementation

For a single observation (i = right slot, j = left slot), with
beta_right the right-slot position offset:

    log lambda_i = theta[rights] + beta_right
    log lambda_j = theta[lefts]
    log sqrt(lambda_i * lambda_j) = 0.5 * (log lambda_i + log lambda_j)

The three log-probabilities (up to a shared log-Z constant) are:

    log P(LEFT wins)   = theta[lefts]
    log P(TIE)         = 0.5 * (theta[lefts] + theta[rights] + beta_right) + eta_tie
    log P(RIGHT wins)  = theta[rights] + beta_right

A custom log-likelihood is added via ``pm.Potential`` rather than
``pm.Categorical`` (the latter is not exposed on every pytensor
build). The categorical log-prob is built by hand and accumulated
with the observed codes.

# Backward compatibility

Legacy 5-level observations (LEFT_STRONG, RIGHT_STRONG) are collapsed
to 3-level (LEFT, RIGHT) before the fit. Collapse is the identity on
LEFT, TIE, RIGHT. Old data on disk loads without migration.

# Sign conventions

- ``theta`` is relative to the current candidate field. ``sum(theta) = 0``
  is enforced at every posterior draw via ``pm.ZeroSumNormal``.
- Larger ``theta`` means stronger in general.
- ``beta_right > 0`` means the right slot is advantaged.
- For position-neutral predictions (e.g. tournament scores that
  shouldn't depend on which slot the item happened to be on), set
  ``beta_right = 0`` by passing ``position_neutral=True`` to
  ``summarize_btd``.

# Identifiability

``theta`` is sum-to-zero; there is no separate intercept. ``eta_tie``
is a single scalar (no per-item tie propensity). ``sigma_theta`` is
the global scale of the strengths. The model has 3 global parameters
plus ``N - 1`` sum-to-zero thetas, totaling ``N + 2`` effective
parameters.
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

from .protocol import Observation, collapse_to_3_level


def _btd_code(verdict: str) -> int:
    """Map a verdict to the BTD 3-level outcome code.

    LEFT_STRONG, LEFT  -> 0 (left wins)
    TIE                -> 1 (tie)
    RIGHT, RIGHT_STRONG -> 2 (right wins)

    Accepts both legacy 5-level and current 3-level verdict strings.
    """
    v = collapse_to_3_level(verdict)
    if v == "LEFT":
        return 0
    if v == "TIE":
        return 1
    if v == "RIGHT":
        return 2
    raise ValueError(f"unknown verdict: {verdict!r}")


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

        # Davidson likelihood in log space.
        #   log lambda_i = theta[rights] + beta_right  (right slot)
        #   log lambda_j = theta[lefts]                  (left slot)
        #   log sqrt(lambda_i * lambda_j) = 0.5 * (log lambda_i + log lambda_j)
        log_pi = theta[rights] + beta_right
        log_pj = theta[lefts]
        log_d = 0.5 * (log_pi + log_pj)

        a = log_pj                              # log P(left wins)   (up to -log Z)
        b = log_d + eta_tie                     # log P(tie)
        c = log_pi                              # log P(right wins)

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

    The 5-level ordinal verdicts are collapsed to 3 outcomes
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

def _position_neutral_beta(beta_right_draws: np.ndarray, position_neutral: bool) -> np.ndarray:
    """Return beta_right draws, optionally forced to zero."""
    if not position_neutral:
        return beta_right_draws
    return np.zeros_like(beta_right_draws)


def summarize_btd(
    result: BTDFitResult,
    observations=None,
    hdi_prob: float = 0.9,
    position_neutral: bool = False,
) -> dict:
    """All posterior summaries in one call. JSON-serializable.

    position_neutral:
        If True, the per-pair and per-item predictions use
        ``beta_right = 0``. The reported ``beta_right_mean`` and HDI
        still come from the original posterior; only the predictions
        are forced neutral. This is the right setting for tournament
        scores and item rankings that should not depend on which slot
        the item happened to appear in.

        The default is ``False`` so users get the full posterior
        summary including position effects. Set to ``True`` when
        using the predictions to rank or score items.

    Keys mirror the M0 summarize() output where comparable, plus
    ``eta_tie`` and ``nu`` for the tie-weight parameter. Pairwise
    keys also include ``p_left_wins`` and ``p_right_wins`` from the
    BTD likelihood directly (which already incorporate tie
    probability).
    """
    theta = result.theta_draws
    S, n = theta.shape
    beta = _position_neutral_beta(result.beta_right_draws, position_neutral)

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
    # the chosen beta_right, possibly forced to zero).
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    pairwise_lh: dict = {}
    if observations is not None:
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

    beta_orig = result.beta_right_draws
    beta_hdi = az.hdi(beta_orig, hdi_prob=hdi_prob)
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
            "beta_right_mean": float(beta_orig.mean()),
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
        "position_neutral": bool(position_neutral),
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
    """Per-item direct W/L/T counts and direct pairwise tallies.

    This is the baseline view, no model. STRONG verdicts in the
    observations are collapsed to ordinary LEFT/RIGHT before counting
    so the tallies are over a 3-outcome alphabet.

    The dedup invariant is: every (a, b) pair appears in both
    orientations. If some orientations are missing, the per-pair
    tallies are still reported but should be interpreted as partial.

    Returns:
        {
          "per_item": {"wins": {...}, "losses": {...}, "ties": {...}},
          "pairwise": {"<a>,<b>": {"wins_first": int, "wins_second": int, "ties": int}},
          "tournament_score": {"<id>": float, ...},  # tie-adjusted, position-neutral
          "n_observations": int,
          "n_left_strong": int,    # how many LEFT_STRONG were collapsed
          "n_right_strong": int,
        }

    tournament_score is a per-item score that gives half credit for
    ties and full credit for wins, normalized by the number of
    other items (N-1). A score of 1.0 means the item won against
    every other item; 0.0 means it lost to every other item. The
    score is position-neutral (it does not depend on which slot the
    item appeared in).
    """
    from collections import defaultdict

    item_wins: dict[str, int] = defaultdict(int)
    item_losses: dict[str, int] = defaultdict(int)
    item_ties: dict[str, int] = defaultdict(int)
    pairs: dict[tuple[str, str], dict[str, int]] = {}

    seen_items: set[str] = set()
    for o in observations:
        if not o.verdict:
            continue
        c = _btd_code(o.verdict)
        seen_items.add(o.a)
        seen_items.add(o.b)
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

    # Tie-adjusted tournament score: wins + 0.5 ties, normalized by
    # the number of other items. This is a position-neutral score.
    n = len(seen_items)
    tournament_score: dict[str, float] = {}
    if n > 1:
        denom = n - 1
        for item in seen_items:
            w = item_wins.get(item, 0)
            t = item_ties.get(item, 0)
            tournament_score[item] = (w + 0.5 * t) / denom

    return {
        "per_item": {
            "wins": dict(item_wins),
            "losses": dict(item_losses),
            "ties": dict(item_ties),
        },
        "pairwise": {
            f"{p[0]},{p[1]}": v for p, v in pairs.items()
        },
        "tournament_score": tournament_score,
        "n_observations": len([o for o in observations if o.verdict]),
        "n_left_strong": sum(1 for o in observations if o.verdict == "LEFT_STRONG"),
        "n_right_strong": sum(1 for o in observations if o.verdict == "RIGHT_STRONG"),
    }
