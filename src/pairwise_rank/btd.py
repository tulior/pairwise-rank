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

These three logits are stacked along the categorical axis and
passed to ``pm.Categorical("y", logit_p=logits, observed=ys)``.
PyMC's categorical primitive owns the softmax / logsumexp / observed
log-prob accumulation, so the fit does not implement any custom
log-softmax or normalization math. The same three logits are
reused verbatim by ``predict_btd`` and the per-pair block in
``summarize_btd``, where they are normalized with
``scipy.special.softmax``. There is no second hand-rolled softmax
site.

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
from scipy.special import softmax

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


# ----------------------------------------------------------------------------
# Vectorized summary helpers
# ----------------------------------------------------------------------------
#
# These helpers own the small amounts of math that were previously
# inlined as Python loops in summarize_btd and predict_btd. They are
# not generic abstractions; they exist to:
#   (a) keep one canonical formula for the Davidson logits, and
#   (b) replace O(n) and O(n^2) per-item Python loops over the
#       posterior draws with vectorized NumPy reductions.
#
# Shapes:
#   theta              (S, n)   flat posterior strengths
#   beta_right_draws   (S,)     flat position-effect draws
#   log_nu             (S,)     log tie-weight draws
#   theta_left, theta_right  (S,) or (S, n_obs)  -- broadcastable
#   beta, log_nu              (S,)                -- broadcastable

def _rank_pos(theta: np.ndarray) -> np.ndarray:
    """Return (n, S) 0-indexed rank positions: rank_pos[i, s] is
    the position of item i in draw s when theta is sorted
    descending. Uses stable argsort tie-breaking (first occurrence
    wins, matching the pre-vectorization contract).
    """
    order = np.argsort(-theta, axis=1)            # (S, n)
    rank_pos = np.empty_like(order)
    S = order.shape[0]
    rank_pos[np.arange(S)[:, None], order] = np.arange(order.shape[1])
    return rank_pos.T                             # (n, S)


def _p_best(theta: np.ndarray) -> np.ndarray:
    """Return (n,) P(best) using the same argmax tie-breaking as
    the pre-vectorization code: each draw credits exactly one
    item (the first-argmax item on ties). Equivalent to
    ``(np.argmax(theta, axis=1) == i).mean()`` per item, but
    vectorized via bincount so we do not loop over n.
    """
    argmax_items = np.argmax(theta, axis=1)         # (S,)
    n = theta.shape[1]
    counts = np.bincount(argmax_items, minlength=n)
    return counts / theta.shape[0]


def _pairwise_gt_means(theta: np.ndarray) -> np.ndarray:
    """Return (n, n) matrix of ``mean(theta_s[i] > theta_s[j])``
    across the posterior. Diagonal is set to NaN to match the
    pre-vectorization contract (the pre-vectorization code used
    ``np.full((n, n), np.nan)`` and never wrote the diagonal).
    """
    pairwise = (theta[:, :, None] > theta[:, None, :]).mean(axis=0)
    np.fill_diagonal(pairwise, np.nan)
    return pairwise


def _davidson_probs(
    theta_left: np.ndarray,
    theta_right: np.ndarray,
    beta: np.ndarray,
    log_nu: np.ndarray,
) -> np.ndarray:
    """Compute BTD probabilities (LEFT, TIE, RIGHT) for any
    broadcastable shape of (theta_left, theta_right). The three
    logits are the canonical Davidson equation:

        log p_left  = theta_left
        log p_tie   = log_nu + 0.5 * (theta_left + theta_right + beta)
        log p_right = theta_right + beta

    Returns the softmax-normalized probabilities along the last
    axis. ``beta`` and ``log_nu`` are (S,) and are aligned with
    the last axis of (theta_left, theta_right); when those are
    2D (S, n) or (S, n_obs) the broadcast works because beta is
    reshaped to (S, 1). This is the only place in the production
    source where the Davidson logits are constructed, so the fit
    and the predict path cannot drift apart.
    """
    # Align beta / log_nu with the last axis of the per-item
    # strength arrays so the (S,) shape broadcasts cleanly against
    # (S,) and (S, n_obs) inputs.
    if theta_left.ndim >= 2:
        beta = beta.reshape((-1,) + (1,) * (theta_left.ndim - 1))
        log_nu = log_nu.reshape((-1,) + (1,) * (theta_left.ndim - 1))
    log_pi = theta_right + beta
    log_pj = theta_left
    log_d = 0.5 * (log_pi + log_pj)
    logits = np.stack([log_pj, log_d + log_nu, log_pi], axis=-1)
    return softmax(logits, axis=-1)


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
    divergences: int | None = None
    """Number of divergent transitions post-warmup across all chains.
    Stored on the result so callers do not need to reach into
    ``idata.sample_stats`` to check sampler health. ``None`` means
    the count could not be read (e.g. the sampler backend does not
    report divergences) and the fit should be treated as
    **unverified for geometry**, not as "0 divergences = healthy"."""

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

        # Davidson likelihood logits. For each observation, the
        # three log-probabilities (up to a shared log-Z) are
        #
        #   log P(LEFT wins)  = theta[lefts]
        #   log P(TIE)        = 0.5 * (theta[lefts] + theta[rights] + beta_right) + eta_tie
        #   log P(RIGHT wins) = theta[rights] + beta_right
        #
        # (right slot is the "i" in the Davidson lambda_i
        # convention; beta_right is the right-slot position
        # offset). The categorical normalization (softmax) is
        # owned by ``pm.Categorical`` itself, so we do not
        # implement a custom log-softmax here. The same three
        # logits are reused verbatim by ``predict_btd`` and the
        # per-pair block in ``summarize_btd``, where they are
        # normalized with ``scipy.special.softmax``.
        log_pi = theta[rights] + beta_right      # right slot
        log_pj = theta[lefts]                    # left slot
        log_d = 0.5 * (log_pi + log_pj)          # geometric mean

        logits = pt.stack([log_pj, log_d + eta_tie, log_pi], axis=1)
        pm.Categorical("y", logit_p=logits, observed=ys)
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

    # Divergences are a sampler health indicator. We expose the count
    # both on the BTDFitResult dataclass and in the summarize_btd
    # output so callers do not have to reach into idata directly.
    # The default for "could not be read" is None, NOT 0: a
    # missing or unrecognised divergences field means we cannot
    # certify sampler geometry, and reporting 0 would make a
    # broken or misconfigured fit look healthy.
    try:
        n_divergences: int | None = int(
            idata.sample_stats["diverging"].sum().item()
        )
    except (KeyError, AttributeError, TypeError):
        n_divergences = None

    return BTDFitResult(
        idata=idata,
        n=n,
        item_ids=list(item_ids),
        divergences=n_divergences,
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

    Keys: per-item (theta_mean, theta_hdi, p_best, p_top2,
    expected_rank), pairwise (p_i_gt_j, delta_mean, delta_hdi,
    and BTD-likelihood p_left_wins / p_tie / p_right_wins when
    observations are provided), position_effect, sigma_theta,
    tie_parameter (eta_tie and nu), sampler diagnostics.
    """
    # Pull all posterior arrays once.
    theta = result.theta_draws                  # (S, n)
    S, n = theta.shape
    beta_orig = result.beta_right_draws         # (S,)
    beta = _position_neutral_beta(beta_orig, position_neutral)
    eta_tie = result.eta_tie_draws              # (S,)
    nu = result.nu_draws                        # (S,)
    log_nu = np.log(nu)                         # (S,)

    # Per-item rank / probability summaries (vectorized). The
    # tie-breaking semantics match the pre-vectorization code:
    # argsort is stable and argmax picks the first-occurrence on
    # ties, so ``P(best)`` is the fraction of draws that credited
    # each item via the per-draw argmax.
    rank_pos = _rank_pos(theta)                 # (n, S), 0-indexed
    p_best = _p_best(theta)                     # (n,)
    p_top2 = (rank_pos <= 1).mean(axis=1)      # (n,)
    mean_rank = rank_pos.mean(axis=1) + 1       # (n,), 1-indexed

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

    # Pairwise P(theta_i > theta_j) -- vectorized via broadcasting.
    # Diagonal is set to NaN to match the pre-vectorization contract.
    P = _pairwise_gt_means(theta)               # (n, n)

    # Per-pair BTD likelihood probabilities. Vectorized across
    # observations (and across the posterior draws) using the
    # canonical Davidson logits; the per-pair aggregation is the
    # only remaining Python loop and is over candidate pairs
    # (O(n_obs), independent of S).
    pairwise_lh: dict = {}
    if observations is not None:
        item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
        left_idx = np.array(
            [item_to_idx[o.left] for o in observations], dtype=int,
        )
        right_idx = np.array(
            [item_to_idx[o.right] for o in observations], dtype=int,
        )
        # (S, n_obs, 3) BTD probabilities for every observation.
        obs_probs = _davidson_probs(
            theta[:, left_idx], theta[:, right_idx], beta, log_nu,
        )
        obs_probs_mean = obs_probs.mean(axis=0)   # (n_obs, 3)
        # Group observations by unordered pair. We sort the
        # (left, right) pair tuples into canonical (min, max) order,
        # then walk the sorted array and close each group on tuple
        # change. This is O(n_obs log n_obs) for the sort and
        # O(n_obs) for the walk; the only remaining Python loop in
        # the per-pair block is over the number of distinct pairs
        # in the observation set, not over posterior draws.
        pair_left = np.minimum(left_idx, right_idx)
        pair_right = np.maximum(left_idx, right_idx)
        order = np.lexsort((pair_left, pair_right))
        sorted_left = pair_left[order]
        sorted_right = pair_right[order]
        sorted_probs = obs_probs_mean[order]
        prev_pair: tuple[int, int] | None = None
        accum = np.zeros(3)
        count = 0
        for k in range(len(observations)):
            pair = (int(sorted_left[k]), int(sorted_right[k]))
            if pair != prev_pair:
                if prev_pair is not None:
                    pairwise_lh[prev_pair] = {
                        "p_left_wins_mean": float(accum[0] / count),
                        "p_tie_mean": float(accum[1] / count),
                        "p_right_wins_mean": float(accum[2] / count),
                    }
                prev_pair = pair
                accum[:] = 0.0
                count = 0
            accum += sorted_probs[k]
            count += 1
        if prev_pair is not None:
            pairwise_lh[prev_pair] = {
                "p_left_wins_mean": float(accum[0] / count),
                "p_tie_mean": float(accum[1] / count),
                "p_right_wins_mean": float(accum[2] / count),
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

    beta_hdi = az.hdi(beta_orig, hdi_prob=hdi_prob)
    sigma = result.sigma_theta_draws
    sigma_hdi = az.hdi(sigma, hdi_prob=hdi_prob)
    eta_tie_hdi = az.hdi(eta_tie, hdi_prob=hdi_prob)
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

    # Sampler diagnostics. divergences is the count of divergent
    # transitions across all chains, or None if the count could not
    # be read (in which case the fit should be treated as
    # unverified for geometry, NOT as "0 divergences = healthy").
    # rhat, ess_bulk, ess_tail are max/min across theta,
    # sigma_theta, eta_tie, beta_right — the scalar parameters
    # plus the per-item theta. A healthy fit has rhat < 1.01 and
    # ess_bulk / ess_tail > ~400. Divergences are fit failures,
    # not cosmetic caveats. A None value is a red flag, not a
    # pass.
    out["divergences"] = getattr(result, "divergences", None)
    try:
        diag_summary = az.summary(
            result.idata,
            var_names=["theta", "sigma_theta", "eta_tie", "beta_right"],
            hdi_prob=hdi_prob,
        )
        out["max_rhat"] = float(diag_summary["r_hat"].max())
        out["min_ess_bulk"] = float(diag_summary["ess_bulk"].min())
        out["min_ess_tail"] = float(diag_summary["ess_tail"].min())
    except Exception:
        # If arviz summary fails for any reason, fall back to None
        # rather than crash the whole summarize call. The values
        # remain inspectable via result.idata.
        out["max_rhat"] = None
        out["min_ess_bulk"] = None
        out["min_ess_tail"] = None

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
# Per-cell (orientation-aware) BTD likelihood predictions
# ----------------------------------------------------------------------------

def predict_btd(
    result: BTDFitResult,
    observations,
    position_neutral: bool = False,
) -> list[dict]:
    """Orientation-aware per-cell BTD likelihood probabilities.

    For each observation in ``observations``, return one dict with the
    BTD likelihood probabilities of LEFT wins / TIE / RIGHT wins,
    averaged over posterior draws. This is the per-cell counterpart
    of the per-unordered-pair averaging inside ``summarize_btd``.

    Use this when you need to know what the model predicts for a
    specific orientation (e.g. audit tables, debugging pairwise
    disagreements, or comparing the same unordered pair at its two
    orientations separately).

    Parameters
    ----------
    result:
        A ``BTDFitResult`` returned by ``fit_btd``.
    observations:
        An iterable of ``Observation`` (or any object with
        ``.left``, ``.right``, ``.verdict``, ``.repeat``). Rows with
        empty verdicts are skipped.
    position_neutral:
        If True, the predictions use ``beta_right = 0``. The reported
        ``beta_right_mean`` and HDI in ``summarize_btd`` still come
        from the original posterior; only the predictions are forced
        neutral.

    Returns
    -------
    list of dict, one per observation (in input order). Each dict has:

        {
          "left": str,
          "right": str,
          "repeat": int,
          "verdict": str,                   # input verdict, may be ""
          "p_left_wins": float,             # P(LEFT wins) under BTD, posterior mean
          "p_tie": float,                   # P(TIE) under BTD, posterior mean
          "p_right_wins": float,            # P(RIGHT wins) under BTD, posterior mean
        }

    Invariants:

        p_left_wins + p_tie + p_right_wins = 1.0  (up to float precision)
        Items not in result.item_ids raise ValueError.
        Legacy 5-level verdicts in the input are accepted; they are
        collapsed to 3-level internally for the fit but the input
        verdict string is preserved in the output.
    """
    item_to_idx = {i: k for k, i in enumerate(result.item_ids)}
    theta = result.theta_draws
    beta_draws = (
        np.zeros_like(result.beta_right_draws)
        if position_neutral
        else result.beta_right_draws
    )
    log_nu = np.log(result.nu_draws)

    out: list[dict] = []
    for o in observations:
        if o.left not in item_to_idx:
            raise ValueError(
                f"item {o.left!r} not in fit.item_ids {result.item_ids!r}"
            )
        if o.right not in item_to_idx:
            raise ValueError(
                f"item {o.right!r} not in fit.item_ids {result.item_ids!r}"
            )
        if not o.verdict:
            continue
        i = item_to_idx[o.left]
        j = item_to_idx[o.right]
        # Same three logits as the fit, via the canonical helper.
        # Categorical normalization is owned by scipy.special.softmax
        # (inside _davidson_probs); no custom log-softmax or
        # max-shift trick is implemented here.
        probs = _davidson_probs(theta[:, i], theta[:, j], beta_draws, log_nu)
        out.append({
            "left": o.left,
            "right": o.right,
            "repeat": o.repeat,
            "verdict": o.verdict,
            "p_left_wins": float(probs[:, 0].mean()),
            "p_tie": float(probs[:, 1].mean()),
            "p_right_wins": float(probs[:, 2].mean()),
        })
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
          "tournament_score": {"<id>": float | None, ...},  # tie-adjusted, position-neutral, in [0, 1]
          "n_observations": int,
          "n_left_strong": int,    # how many LEFT_STRONG were collapsed
          "n_right_strong": int,
        }

    tournament_score is a per-item score that gives half credit for
    ties and full credit for wins, normalized by the *observed*
    total number of judgments for that item:

        score_i = (W_i + 0.5 * T_i) / (W_i + L_i + T_i)

    where W_i, L_i, T_i are the item's win / loss / tie counts
    across the full observation set (every orientation and every
    repeat is counted). Range is [0, 1]:

        all wins    -> 1.0
        all losses  -> 0.0
        all ties    -> 0.5
        mixed       -> strictly between 0 and 1

    The score is position-neutral: it does not depend on which slot
    the item appeared in, only on the verdicts. For a complete
    balanced tournament with both orientations and K repeats per
    orientation, the observed denominator equals 2*K*(N-1) and the
    score is equivalent to (W + 0.5*T) / (2*K*(N-1)). The observed
    form is preferred because it is well-defined on incomplete,
    resumed, or filtered data sets where some orientations or
    repeats are missing.

    Items with no observations (defensive: cannot happen under the
    current invariants, since seen_items is built only from
    observations with verdicts) are reported with a score of
    None -- the package convention for an unavailable per-item
    summary, matching ``out["max_rhat"] = None`` in
    :func:`summarize_btd` when ArviZ diagnostics cannot be
    computed.
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

    # Tie-adjusted tournament score: half credit for ties, full
    # credit for wins, normalized by the *observed* total
    # judgments for the item (W + L + T). This gives a
    # probability-like score in [0, 1] and is robust to
    # incomplete / resumed / filtered data sets. The position-
    # neutrality of the verdict counts is preserved here, so the
    # score is position-neutral by construction.
    tournament_score: dict[str, float | None] = {}
    for item in seen_items:
        w = item_wins.get(item, 0)
        l = item_losses.get(item, 0)
        t = item_ties.get(item, 0)
        denom = w + l + t
        tournament_score[item] = (w + 0.5 * t) / denom if denom > 0 else None

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
