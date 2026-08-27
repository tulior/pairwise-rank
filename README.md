# pairwise-rank

Small research tool for repeated balanced pairwise ordinal judgments with uncertainty-aware ranking.

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
fits the default model, and writes a summary to
`/tmp/pairwise_rank_synthetic/fit_summary.json`.

The standard flow:

```python
from pairwise_rank import (
    run_tournament, save_observations_jsonl, load_observations_jsonl,
    fit, summarize, posterior_predictive_check,
)

# 1. Build a judge function (left_id, right_id) -> verdict
def my_judge(left, right):
    # call your model, return one of the five verdict strings
    ...

# 2. Run the tournament; observations are returned with verdicts filled in
candidates = ["alpha", "beta", "gamma", "delta"]
observations = run_tournament(candidates, my_judge, repeats=3)

# 3. Persist
save_observations_jsonl("observations.jsonl", observations)

# 4. Fit
result = fit(load_observations_jsonl("observations.jsonl"))

# 5. Summarize
summary = summarize(result, observations)
print(summary["per_item"])          # theta, P(best), expected_rank
print(summary["pairwise"])          # P(theta_i > theta_j)
print(summary["position_effect"])   # beta_right mean and HDI

# 6. Optional: one-shot posterior predictive check
ppc = posterior_predictive_check(result, observations)
```

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
ids), `repeat` (1-based index), `verdict` (one of the five labels), and
`reasoning` (optional free-form audit metadata, e.g. the model's
reasoning text). Rows are never averaged. Storage is JSON Lines, one
row per line. The deduplication key is `(a, b, left, right, repeat)`;
`reasoning` is not part of the key.

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

```
eta = theta_right - theta_left + beta_right
theta ~ ZeroSumNormal(sigma = sigma_theta)        # sum-to-zero
sigma_theta ~ HalfNormal(1.0)
beta_right ~ Normal(0, 0.5)                       # right-slot position effect
cutpoints: 3 positive gaps via softplus(gap_raw), then zero-centered
y_obs ~ OrderedLogistic(eta, cutpoints)          # 0..4
```

Sign conventions:

- Larger `theta` means stronger in general.
- `beta_right > 0` means the right slot is advantaged.
- Verdict scale: 0 = `LEFT_STRONG`, 1 = `LEFT`, 2 = `TIE`, 3 = `RIGHT`, 4 = `RIGHT_STRONG`.
- `P(left wins) = P(y in {0,1}) = sigmoid(c_1 - eta)` using the upper bound of the LEFT region.
- `P(TIE) = P(y = 2) = sigmoid(c_2 - eta) - sigmoid(c_1 - eta)`.

The default model has one global strength per item and one
right-slot position effect. It does not include cycle-space or
per-cell random effects; those are experimental extensions that
are not part of v0.1.

## Results

`summarize(result, observations)` returns:

- `per_item`: per-item `theta_mean`, 90% HDI, `P(best)`, `P(top2)`, `expected_rank`.
- `pairwise`: `P(theta_i > theta_j)` and posterior delta HDI for every unordered pair.
- `position_effect`: `beta_right` mean and 90% HDI.
- `sigma_theta`: posterior of the scale of the global strengths.
- `cutpoints`: posterior of the four ordered cutpoints.
- `verdict_distribution`: counts of each verdict label in the observations.

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

## License

MIT. See `LICENSE`.
