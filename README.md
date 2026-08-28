# pairwise-rank

Small research tool for repeated balanced pairwise ranking with uncertainty quantification.

The core is **win/tie/loss** ranking with position bias and Bayesian
global inference. STRONG verdicts are an optional 5-level extension
preserved for backward compatibility; the default is the 3-level
scale `LEFT, TIE, RIGHT`.

## Architecture

```
direct_summary     baseline / always (no model, raw W/L/T)
fit_btd            default probabilistic model (3-level BTD)
fit_ordinal        optional / legacy (5-level ordered logit)
fit                DEPRECATED alias for fit_ordinal
```

Why BTD is the default: across many tournaments (textual, bio,
prefix-match, hex batch, 0x class), STRONG verdicts occur in
≤2% of observations, and BTD vs the 5-level ordered logit give
`r_θ > 0.99` and `r_P(best) > 0.99`. The 3-level protocol is simpler
for the judge, has fewer tool-schema errors, and matches observed
behavior more closely. Same decision information, simpler likelihood,
prefer simpler.

Use `fit_ordinal` only when:
- STRONG responses occur often enough to matter
  (a rough trigger: STRONG > 10-15% of non-ties);
- STRONG vs ordinary wins show demonstrably different behavior;
- the prompt deliberately elicits intensity;
- BTD and direct evidence show unresolved structure that the
  ordinal information might explain;
- you are specifically studying whether preference magnitude
  matters.

## Usage

Install:

```
pip install -e .
```

Run the synthetic example:

```
python examples/synthetic.py
```

This writes observations to `/tmp/pairwise_rank_synthetic/observations.jsonl`,
fits the default BTD model, and writes a summary to
`/tmp/pairwise_rank_synthetic/fit_summary.json`.

The standard flow:

```python
from pairwise_rank import (
    run_tournament, save_observations_jsonl, load_observations_jsonl,
    fit_btd, summarize_btd, direct_summary,
)

# 1. Build a judge function (left_id, right_id) -> verdict
def my_judge(left, right):
    # call your model, return one of: "LEFT", "TIE", "RIGHT"
    ...

# 2. Run the tournament; observations are returned with verdicts filled in
candidates = ["alpha", "beta", "gamma", "delta"]
observations = run_tournament(candidates, my_judge, repeats=3)

# 3. Persist
save_observations_jsonl("observations.jsonl", observations)

# 4. Direct (model-free) baseline
direct = direct_summary(observations)
print(direct["per_item"])  # wins, losses, ties

# 5. Fit BTD (default probabilistic model)
result = fit_btd(load_observations_jsonl("observations.jsonl"))

# 6. Summarize
summary = summarize_btd(result, observations)
print(summary["per_item"])          # theta, P(best), expected_rank
print(summary["pairwise"])          # P(theta_i > theta_j)
print(summary["position_effect"])   # beta_right mean and HDI
print(summary["tie_parameter"])     # nu (Rao-Kupper tie weight)
```

For 5-level ordinal data on disk (legacy), `fit_btd` automatically
collapses `LEFT_STRONG → LEFT` and `RIGHT_STRONG → RIGHT`. To use
the 5-level ordered-logit model explicitly, use `fit_ordinal` and
`run_tournament(..., verdict_levels=VERDICT_LEVELS_5)`. See
`examples/three_view.py` and the test_recovery.py test for the
5-level path.

Resume is achieved by passing existing observations back in:

```python
done = load_observations_jsonl("observations.jsonl")
more = run_tournament(candidates, my_judge, repeats=3, existing=done)
save_observations_jsonl("observations.jsonl", done + more)
```

## Protocol

The protocol is generic and combinatorial. It enumerates every unordered
pair, expands each into both orientations, and produces K independent
repeats per oriented cell. The schedule is deterministic given the
input candidate list and repeats.

The judging instruction defines the quantity being estimated.
Changing the instruction can change the ranking because it changes the
evaluation construct. Reproducible judgments do not establish that the
chosen construct is appropriate.

Each observation row has seven fields: `a`, `b` (the canonical unordered
pair in original candidate-list order), `left`, `right` (the displayed
ids), `repeat` (1-based index), `verdict` (one of the 3-level labels
by default; 5-level labels if observed on legacy data), and `reasoning`
(optional free-form audit metadata, e.g. the model's reasoning text).
Rows are never averaged. Storage is JSON Lines, one row per line.
The deduplication key is `(a, b, left, right, repeat)`; `reasoning`
is not part of the key.

The package does not provide a default prompt, judge, or LLM tool
schema. Those are the caller's job. The package owns the schedule,
the verdict vocabulary, and the model. Nothing else.

### Optional: storing reasoning traces

A `judge_fn` can return either a `Verdict` string or a
`(Verdict, reasoning_str)` tuple. The reasoning string is stored on
each `Observation` as audit metadata and is preserved through
JSONL save/load, but it is ignored by the ranking model. Rows
written before this field existed load with an empty string.

```python
def my_judge(left, right):
    verdict, reasoning = call_my_model(left, right)
    return (verdict, reasoning)

observations = run_tournament(candidates, my_judge, repeats=3)
save_observations_jsonl("observations.jsonl", observations)

# reasoning lives on each observation but never enters the fit
for o in load_observations_jsonl("observations.jsonl"):
    print(o.verdict, "—", o.reasoning[:80])
```

## Model

Two models are provided. The default is **BTD** (3-level); the
**5-level ordered logit** is preserved as `fit_ordinal` for cases
where intensity matters.

### BTD (default, 3-level)

```
eta = theta_right - theta_left + beta_right
theta ~ ZeroSumNormal(sigma = sigma_theta)        # sum-to-zero
sigma_theta ~ HalfNormal(1.0)
beta_right ~ Normal(0, 0.5)                       # right-slot position effect
eta_tie ~ Normal(0, 1)                            # log Rao-Kupper tie weight
nu = exp(eta_tie)                                 # tie weight

log P(left wins)  = theta[ left] - Z
log P(tie)        = 0.5*(theta[ left] + theta[ right] + beta_right) + eta_tie - Z
log P(right wins) = theta[ right] + beta_right - Z
y_obs ~ Categorical(softmax([left, tie, right]))
```

Sign conventions:

- Larger `theta` means stronger in general.
- `beta_right > 0` means the right slot is advantaged.
- `nu = 1` is the symmetric tie prior; `nu > 1` favors ties, `nu < 1` penalizes them.

### Ordered logistic (optional / legacy, 5-level)

```
eta = theta_right - theta_left + beta_right
theta ~ ZeroSumNormal(sigma = sigma_theta)        # sum-to-zero
sigma_theta ~ HalfNormal(1.0)
beta_right ~ Normal(0, 0.5)                       # right-slot position effect
cutpoints: 3 positive gaps via softplus(gap_raw), then zero-centered
y_obs ~ OrderedLogistic(eta, cutpoints)          # 0..4
```

- Verdict scale: 0 = `LEFT_STRONG`, 1 = `LEFT`, 2 = `TIE`, 3 = `RIGHT`, 4 = `RIGHT_STRONG`.
- `P(left wins) = P(y in {0,1}) = sigmoid(c_1 - eta)` using the upper bound of the LEFT region.
- `P(TIE) = P(y = 2) = sigmoid(c_2 - eta) - sigmoid(c_1 - eta)`.

Use only when the 5-level scale is actually carrying useful information
(see the Architecture section for triggers).

## Results

`summarize_btd(result, observations)` returns:

- `per_item`: per-item `theta_mean`, 90% HDI, `P(best)`, `P(top2)`, `expected_rank`.
- `pairwise`: `P(theta_i > theta_j)`, posterior delta HDI, and the
  Rao-Kupper likelihood probabilities `P(left wins)`, `P(tie)`,
  `P(right wins)` averaged across orientations.
- `position_effect`: `beta_right` mean and 90% HDI.
- `sigma_theta`: posterior of the scale of the global strengths.
- `tie_parameter`: `eta_tie` and `nu` posterior summaries.
- `verdict_distribution_btd`: collapsed 3-level counts.

Interpretation:

- `theta` is relative to the current field. Sum-to-zero is enforced
  within the items being fit, not against any external reference.
- `P(best)` is a joint event over the field, not a normalized softmax
  of posterior means. A flat field with high uncertainty can have
  `P(best)` around `1/n` for every item.
- Position-neutral predictions: set `beta_right = 0` for predictions
  if you do not want to assume a right-slot advantage.
- Repeats are separate observations. The model is not given knowledge
  of within-cell correlation.

## Diagnostics

- `posterior_predictive_check(result, observations)`: a single
  repeat-agreement PPC. If observed agreement is in the extreme
  tail of the predictive distribution, the model is underpredicting
  within-cell dependence. A single statistic does not establish
  full calibration; use as one diagnostic, not a certificate.
- Position bias: a non-zero `beta_right` with a 90% HDI excluding
  zero suggests the judge has a left or right slot preference.
- Sampler divergences are failures. Increase `target_accept`,
  lengthen `tune`, or reparameterize. Increasing `K` does not fix
  geometry.

## Three-view report (direct + BTD + ordinal)

For multi-candidate tournaments (≥5 items, ≥30 obs) the routine
report pattern is to compare the three views. BTD is the default
probabilistic model; the 5-level ordered logistic is included as
a cross-check to confirm the ranking is robust to modeling choice.

```python
from pairwise_rank import three_view_report, print_three_view

# include_ordinal=True (default) runs both BTD and the ordered logit.
# Pass include_ordinal=False to skip the M0 fit (saves time when
# STRONG is rare and the cross-check is not informative).
report = three_view_report(observations, draws=2000, tune=2500, chains=4)
print_three_view(report, label="my tournament")

# report["top1"] = {"direct": ..., "btd": ..., "m0": ..., "all_three_agree": ...}
# report["theta_corr_btd_m0"] = Pearson r between BTD and M0 theta means
# report["pbest_corr_btd_m0"] = Pearson r between BTD and M0 P(best) values
```

If all three views agree on top-1, the winner is robust to modeling
choice. If they disagree, the disagreement is diagnostic. In
practice, the BTD vs ordinal-logit correlation is > 0.99 on every
tournament we have run; the cross-check is included as insurance,
not because we expect disagreement. Head-to-heads (≤2 items) are
not informative under either model; use `direct_summary` alone
in that case.

See `examples/three_view.py` for a self-contained reproducible
demonstration.

## License

MIT. See `LICENSE`.
