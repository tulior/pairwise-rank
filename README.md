# pairwise-rank

Small tools for reproducible pairwise win/tie/loss evaluation, direct
summaries, and Bayesian Davidson ranking.

The default protocol is the 3-level scale `LEFT, TIE, RIGHT`. The
historical 5-level scale (`LEFT_STRONG, LEFT, TIE, RIGHT,
RIGHT_STRONG`) is preserved for backward compatibility: legacy
data on disk loads without migration, and `fit_btd` collapses
`STRONG` outcomes into ordinary wins/losses automatically. No
5-level inference is performed.

## Architecture

```
direct_summary     baseline / always (no model, raw W/L/T + tournament score)
fit_btd            default probabilistic model (3-level Bradley-Terry-Davidson)
```

The supported methodology is BTD on 3-level verdicts. The 5-level
ordinal information is rarely used in practice: across many
tournaments, STRONG verdicts occur in roughly 1-2% of non-ties,
and collapsing them to ordinary wins/losses loses almost no
information. New code uses the 3-level protocol exclusively; legacy
data with STRONG verdicts is accepted as input and collapsed
automatically.

## Install

```
pip install -e .
```

## Quickstart

```python
from pairwise_rank import (
    run_tournament, save_observations_jsonl, load_observations_jsonl,
    fit_btd, summarize_btd, direct_summary,
)

# 1. Build a judge function: (left_id, right_id) -> "LEFT" | "TIE" | "RIGHT"
def my_judge(left, right):
    # call your model
    return "LEFT"

# 2. Run the tournament; observations come back with verdicts filled in.
candidates = ["alpha", "beta", "gamma", "delta"]
observations = run_tournament(candidates, my_judge, repeats=3)
save_observations_jsonl("observations.jsonl", observations)

# 3. Direct (model-free) baseline
direct = direct_summary(load_observations_jsonl("observations.jsonl"))
print(direct["per_item"])         # wins / losses / ties
print(direct["tournament_score"]) # tie-adjusted, position-neutral

# 4. Fit BTD (default probabilistic model)
result = fit_btd(load_observations_jsonl("observations.jsonl"))

# 5. Summarize (position_neutral=True for ranking/score, default
#    False for the full posterior summary including beta_right).
summary = summarize_btd(result, observations)
print(summary["per_item"])         # theta, P(best), expected_rank
print(summary["pairwise"])         # P(theta_i > theta_j)
print(summary["position_effect"])  # beta_right mean and HDI
print(summary["tie_parameter"])    # nu (Davidson tie weight)
```

Run the synthetic example:

```
python examples/synthetic.py
```

This writes observations to
`/tmp/pairwise_rank_synthetic/observations.jsonl`, fits the
default BTD model, and writes a summary to
`/tmp/pairwise_rank_synthetic/fit_summary.json`.

Resume a partial run:

```python
done = load_observations_jsonl("observations.jsonl")
more = run_tournament(candidates, my_judge, repeats=3, existing=done)
save_observations_jsonl("observations.jsonl", done + more)
```

## Protocol

The protocol is generic and combinatorial. It enumerates every
unordered pair, expands each into both orientations, and produces
K independent repeats per oriented cell. The schedule is
deterministic given the input candidate list and repeats.

The judging instruction defines the quantity being estimated.
Changing the instruction can change the ranking because it changes
the evaluation construct. Reproducible judgments do not establish
that the chosen construct is appropriate.

Each observation row has seven fields: `a`, `b` (the canonical
unordered pair in original candidate-list order), `left`, `right`
(the displayed ids), `repeat` (1-based index), `verdict` (one of
the 3-level labels by default; 5-level labels if observed on legacy
data), and `reasoning` (optional free-form audit metadata, e.g.
the model's reasoning text). Rows are never averaged. Storage is
JSON Lines, one row per line. The deduplication key is
`(a, b, left, right, repeat)`; `reasoning` is not part of the
key, never enters the fit, and never affects deduplication.

The package does not provide a default prompt, judge, or LLM tool
schema. Those are the caller's job. The package owns the schedule,
the verdict vocabulary, and the model. Nothing else.

### Optional: storing reasoning traces

A `judge_fn` can return either a `Verdict` string or a
`(Verdict, reasoning_str)` tuple. The reasoning string is stored
on each `Observation` as audit metadata and is preserved through
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

## Models

### BTD (default, 3-level)

The BTD model is the Davidson (1970) extension of Bradley-Terry,
with a tie term proportional to `nu * sqrt(lambda_i * lambda_j)`
and `nu > 0`:

```
lambda_i = exp(theta[i] + beta_right_if_i_on_right)
lambda_j = exp(theta[j])

P(i beats j)    = lambda_i / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
P(i ties j)     = nu * sqrt(lambda_i * lambda_j) / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
P(i loses to j) = lambda_j / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
```

Priors:

```
theta        ~ ZeroSumNormal(sigma = sigma_theta)   # sum-to-zero
sigma_theta  ~ HalfNormal(1.0)
beta_right   ~ Normal(0, 0.5)                      # right-slot position effect
eta_tie      ~ Normal(0, 1)                       # log tie weight
nu           = exp(eta_tie)                        # tie weight, > 0
```

Sign conventions:

- Larger `theta` means stronger in general.
- `beta_right > 0` means the right slot is advantaged.
- `nu = 1` is the symmetric tie prior; `nu > 1` favors ties, `nu < 1` penalizes them.
- The likelihood is symmetric in `(i, j)` and reduces to
  Bradley-Terry as `nu -> 0`.

This is **not** Rao-Kupper. Rao-Kupper uses a tie term of the
form `nu * (lambda_i + lambda_j) / 2`; Davidson uses the
geometric-mean form. The two are statistically distinguishable on
real data. We use Davidson.

## Results

`summarize_btd(result, observations, position_neutral=False)` returns:

- `per_item`: per-item `theta_mean`, 90% HDI, `P(best)`, `P(top2)`, `expected_rank`.
- `pairwise`: `P(theta_i > theta_j)`, posterior delta HDI, and the
  Davidson likelihood probabilities `P(left wins)`, `P(tie)`,
  `P(right wins)` averaged across orientations.
- `position_effect`: `beta_right` mean and 90% HDI.
- `sigma_theta`: posterior of the scale of the global strengths.
- `tie_parameter`: `eta_tie` and `nu` posterior summaries.
- `verdict_distribution_btd`: collapsed 3-level counts.
- `divergences`, `max_rhat`, `min_ess_bulk`, `min_ess_tail`:
  sampler health from the post-warmup draws. `divergences` is the
  number of divergent transitions across all chains, or `None` if
  the count could not be read; the other three are arviz-computed
  convergence diagnostics over `theta`, `sigma_theta`, `eta_tie`,
  `beta_right`. A healthy fit has `max_rhat < 1.01` and
  ESS > ~400. A `None` divergence count is a red flag, not a pass.
- `position_neutral`: whether `beta_right` was forced to zero in
  the predictions.

Interpretation:

- `theta` is relative to the current field. Sum-to-zero is
  enforced within the items being fit, not against any external
  reference.
- `P(best)` is a joint event over the field, not a normalized
  softmax of posterior means. A flat field with high uncertainty
  can have `P(best)` around `1/n` for every item.
- `P(best)` is not objective quality. It is a posterior event
  over the current candidate field, model, prompt, and judge.
- Position-neutral predictions: pass `position_neutral=True` to
  `summarize_btd` to set `beta_right = 0` for the predictions.
  Use this when ranking or scoring items. The default
  `position_neutral=False` keeps the position effect in the
  predictions and is the right choice for inspecting the
  full posterior summary.
- Repeats are separate observations. The model is not given
  knowledge of within-cell correlation.

## Per-cell (orientation-aware) predictions

`summarize_btd` reports per-pair likelihood probabilities averaged
across orientations. For orientation-aware audit tables, debugging
pairwise disagreements, or per-row prediction tables, use
`predict_btd`:

```python
from pairwise_rank import predict_btd, fit_btd, load_observations_jsonl

result = fit_btd(load_observations_jsonl("observations.jsonl"))
preds = predict_btd(result, observations, position_neutral=True)
# preds is a list of dict, one per observation (in input order):
#   {"left", "right", "repeat", "verdict",
#    "p_left_wins", "p_tie", "p_right_wins"}
# p_left_wins + p_tie + p_right_wins = 1.0
```

`predict_btd` is the per-cell counterpart of the per-unordered-pair
averaging inside `summarize_btd`. It is the right tool when the
question is "what does the model predict for this specific
(left, right) row?" rather than "what does the model predict for
this unordered pair on average?".

## Diagnostics

- Position bias: a non-zero `beta_right` with a 90% HDI excluding
  zero suggests the judge has a left or right slot preference.
  An HDI that includes zero does not prove there is no bias; it
  means the data is not powerful enough to detect one.
- Sampler divergences are failures. A `divergences` value of
  `None` is also a failure -- it means the sampler backend did
  not report a divergences field and the fit should be treated as
  unverified for geometry. Increase `target_accept`, lengthen
  `tune`, or reparameterize. Increasing `K` does not fix geometry.
- TIE rate is diagnostic. A 50% TIE rate on a construct that
  should produce clear winners suggests the prompt is asking
  the model to hedge. A 0% TIE rate is a signal that the model
  is forced to vote. Neither extreme is itself a quality
  criterion.

## Head-to-head vs multi-candidate

For a two-candidate counterbalanced experiment, direct evidence
is normally the primary report. PyMC is barely identifiable with
two items and small K, and divergent fits must be discarded.
The 3-level verdict scale and direct_summary alone are enough to
characterize the comparison.

For multi-candidate tournaments (≥5 items), use direct + BTD.
Disagreement between the two is diagnostic, not a failure.

## See also

[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) — generic lessons on
construct validity, selection leakage, eligibility vs ranking
dimensions, prompt-design heuristics, and the inference policy
this library is built around.

## License

MIT. See `LICENSE`.
