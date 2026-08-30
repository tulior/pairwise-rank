# AGENTS.md

Operational guide for any coding agent (Mavis, Mavis, or otherwise)
working in this repository. Read this file first.

The companion document `EXPERIMENT_DESIGN.md` holds the design
methodology (constructs, prompt design, campaign shape, scaling
boundary). `AGENTS.md` holds the operating rules for code
maintenance and the statistical/probabilistic contract this
library implements. Both must be respected.

---

## 1. Project purpose

`pairwise-rank` is a small reusable library for reproducible
pairwise ranking under the LEFT / TIE / RIGHT protocol.

The currently supported statistical stack is:

```
LEFT / TIE / RIGHT judgments
-> direct counterbalanced summaries
-> Bayesian Davidson / Bradley-Terry-Davidson
-> optional right-position effect
```

Legacy compatibility is **ingest-only** for the 5-level scale:

```
LEFT_STRONG  -> LEFT
RIGHT_STRONG -> RIGHT
```

This collapse is performed by the protocol layer when loading
observations and by `fit_btd` / `direct_summary` when ingesting
verdicts. The 3-level scale carries all the model structure. No
5-level inference is performed.

M0 (ordered-logistic / 5-level) inference is **removed**. Do
not resurrect it. There is no `fit_ordinal`. There is no
`FitResult`. The methods in the public API are `fit_btd`,
`summarize_btd`, `predict_btd`, `direct_summary`, plus the
protocol helpers.

---

## 2. Design philosophy

The single most important rule in this project:

```
Do not increase model complexity to solve an experimental-design problem.
```

A scaling problem in the design layer (the candidate set is too
large, the comparison budget is bounded, the decision is only
over the top-k) is addressed by changing the design. It is not
addressed by replacing the ranking likelihood with a more
expressive model. A misspecification problem in the model layer
is addressed by changing the model, not by adding richer
machinery to absorb design choices.

The two layers have different jobs:

```
likelihood pools evidence
design buys evidence
```

For the current small-N use case, complete counterbalanced
round robin is the default design. Adaptive pair selection is a
separate future design concern, **not** part of `fit_btd`. See
`EXPERIMENT_DESIGN.md` §20 for the scaling boundary and the
frozen modeling layer for the current use case.

---

## 3. Statistical ownership

This library is intentionally small. It does not reimplement
operations a maintained library already owns.

| library       | owns                                                |
|---------------|-----------------------------------------------------|
| PyMC          | model construction and NUTS sampling                |
| ArviZ         | R-hat, ESS, HDI, MCMC diagnostics, summaries        |
| SciPy         | numerical softmax (used outside PyMC)               |
| `pairwise-rank` | Davidson model equation, direct W/L/T summaries, posterior rank reductions, position-neutral pair predictions, protocol/provider glue |

`pairwise-rank` does **not** own:

- a custom sampler, HMC, or NUTS implementation
- a custom R-hat, ESS, or HDI implementation
- a custom generic softmax outside PyMC
- a custom generic categorical likelihood
- a custom divergence detector
- a custom adaptation scheme

If a question can be answered by a maintained library, use the
maintained library. Before adding machinery, ask:

```
can a maintained library already do this?
```

Prefer deletion over abstraction.

---

## 4. Davidson parameterization (exact)

This is the parameterization `fit_btd` implements. It is
documented here so a future agent does not accidentally drift.

```
sigma_theta ~ HalfNormal(1.0)
theta       ~ ZeroSumNormal(sigma=sigma_theta, shape=n)   # sum-to-zero
beta_right  ~ Normal(0.0, 0.5)
eta_tie     ~ Normal(0.0, 1.0)
nu          = exp(eta_tie)                               # > 0

a_left  = theta[left]
a_right = theta[right] + beta_right

logit P(LEFT)  = a_left
logit P(TIE)   = eta_tie + 0.5 * (a_left + a_right)
logit P(RIGHT) = a_right
```

Sign conventions:

- Larger `theta` means stronger in general.
- `beta_right > 0` means the right slot is advantaged.
- `nu = 1` is the symmetric tie prior; `nu > 1` favors ties;
  `nu < 1` penalizes them.
- The likelihood is symmetric in `(i, j)` and reduces to
  Bradley-Terry as `nu -> 0`.
- This is **Davidson**, not Rao-Kupper. The tie term uses the
  geometric mean `nu * sqrt(lambda_i * lambda_j)`, not
  `nu * (lambda_i + lambda_j) / 2`. The two are statistically
  distinguishable on real data; do not substitute one for the
  other.

Position-neutral prediction (used by `predict_btd(...
position_neutral=True)` and the per-pair probabilities in
`summarize_btd`) **forces `beta_right = 0`** in the prediction
step. The fitted posterior is unchanged; only the prediction
uses the position-neutral edge advantage.

`theta` is **relative to the current candidate field**, not
against any external reference. Sum-to-zero is enforced within
the items being fit. Do not compare `theta` magnitudes across
separately fitted fields.

---

## 5. Evidence interpretation

Direct W/L/T remains the primary raw evidence.

For small matched A/B experiments, prefer direct counterbalanced
tallies. The BTD posterior is barely identifiable with two
items and small K.

Direct tournament score (per item):

```
S_i = (W_i + 0.5 * T_i) / (W_i + L_i + T_i)
```

Range `[0, 1]`, position-neutral, well-defined on incomplete
data:

- all wins   -> 1.0
- all losses -> 0.0
- all ties   -> 0.5
- mixed      -> strictly between 0 and 1

BTD is the global transitive approximation. It pools evidence
across all pairs of items.

Direct-vs-global disagreement is called:

```
direct-vs-global strain
```

Do **not** label every strain a cycle. A cycle requires
demonstrated cyclic structure across at least three candidates.
The model falsification audit at
`experiments/model_falsification/MODEL_FALSIFICATION.md`
documents the empirical evidence that an identified cycle
augmentation does not earn its complexity at the project scale
and that strain-like residuals are typically tie-model
misspecification, not non-transitivity.

---

## 6. Diagnostics

- **Divergences** are model-geometry warnings. A `None`
  divergence count is also a warning, not a pass — it means
  the sampler backend did not report a divergences field and
  the fit should be treated as unverified for geometry.
  Increasing K or `target_accept` does not repair bad posterior
  geometry; it papers over it. If the geometry is broken, the
  fix is in the model or the data, not in sampler settings.
- **R-hat near 1** is necessary, not sufficient. Four chains
  agreeing at R-hat = 1.0 says the chains mixed to the same
  distribution. It does not say that distribution is the right
  one for the data. Inspect per-parameter posteriors.
- **ESS bulk and ESS tail** measure effective Monte Carlo
  information, not statistical coverage. Bulk ESS for central
  summaries, tail ESS for quantiles / HDI endpoints. A chain
  with R-hat = 1.0 and ESS = 50 has not explored enough.
- **One PPC statistic does not prove calibration.** A model can
  match a single chosen summary while being wrong elsewhere.
- **TIE rate is a diagnostic, not a quality score.** A high
  TIE rate may mean the construct does not discriminate, or
  that candidates are genuinely equivalent. A 0% TIE rate
  means the judge differentiated the presented alternatives; it
  does not mean the experiment was intrinsically good.

---

## 7. Testing

The full test suite runs the four statistical files plus two
provider files:

```
tests/test_btd.py             (BTD model invariants, vectorized helpers)
tests/test_btd_predict.py    (per-cell BTD likelihood predictions)
tests/test_protocol.py        (schedule, dedup, verdict collapse, JSONL)
tests/test_v04.py             (default 3-level + legacy 5-level collapse)
tests/test_providers/test_base.py      (connector base types)
tests/test_providers/test_MiniMax.py   (MiniMax connector contract)
```

In execution-time-constrained environments, run the four
statistical files in 4 sequential batches of approximately 2
minutes each. Provider tests are fast (<10 s total) and can
be run in a single command at any time.

```
PYTHONPATH=src pytest tests/test_btd.py
PYTHONPATH=src pytest tests/test_btd_predict.py
PYTHONPATH=src pytest tests/test_protocol.py
PYTHONPATH=src pytest tests/test_v04.py
PYTHONPATH=src pytest tests/test_providers
```

PyMC sampling dominates wall time. Do not parallelize across
files; PyMC already parallelizes chains internally.

Provider tests mock the HTTP layer
(`pairwise_rank.providers.MiniMax.MiniMaxJudge` accepts a
`http_post=...` constructor argument for test injection)
and do not require network access. Live provider integration
tests are opt-in and **not** part of the default suite.

---

## 8. Repository editing rules

Keep this project small.

Before adding machinery ask:

```
can a maintained library already do this?
```

Prefer deletion over abstraction. Do not add:

- registries
- factories
- base-class hierarchies
- plugin frameworks
- sampler wrappers
- capability negotiation
- dependency injection containers
- generic HTTP SDKs
- provider framework code larger than the problem requires

unless multiple real implementations force them. Right now,
there is exactly one provider (MiniMax). A single module is
enough.

Do not add code, examples, or artifacts that belong in
`/workspace/` (the user's private experiment directory):

- LLM runners, experiment drivers, prompts
- API credentials, tokens, or environment-variable reads (the
  one exception is the MiniMax connector, which reads
  `MINIMAX_API_KEY` from the environment)
- Test fixtures containing real LLM responses
- ZIP archives of "experiment runs"
- Markdown reports of experiment results
- Candidate sets, vote tallies, per-cell observation data
- Reasoning traces as statistical input

If a user requests something in the above list, it goes in
`/workspace/`, not in the library.

---

## 9. Provider philosophy

The library ships a **thin MiniMax connector** as a
convenience. It is not a provider framework. There is one
connector module; the abstraction is one Protocol.

The policy for any provider:

1. **Use the provider's default/recommended sampling
   behavior.** Do not hand-tune temperature, top_p, max_p,
   penalties, seed, beam/search controls. The library never
   sends these unless the provider API requires one of them.
2. **Request the maximum supported reasoning explicitly,
   where the API exposes one.** For MiniMax M3, this is
   `reasoning: {"effort": "high"}`. The compatibility values
   `high / medium / low / minimal` enable Adaptive Thinking
   but do not tune its depth. Use the maximum to enable
   reasoning; do not interpret the value as a depth knob.
3. **Centralize the model identifier in the connector.**
   Update it in one place when the model changes.
4. **Authentication from environment, never hard-coded.**
   Document the env var name in the connector docstring.
5. **Preserve enough provider/model metadata for experiment
   reproducibility.** The judgment object should record the
   provider, model, and reasoning effort used.

If temperature / top_p are not sent, record them conceptually
as "provider default / not overridden", not as guessed numeric
values. The connector does not pretend to know the provider's
internal defaults.

---

## 10. Adding a provider

To add a new provider, an agent should:

1. Read the provider's **current official API documentation**.
   The model identifier, request field names, and reasoning
   controls change. Do not rely on memory or older issues.
2. Implement the existing connector interface:
   `judge(JudgmentRequest) -> Judgment`. Do not invent a
   parallel protocol.
3. Keep provider code in its own module under
   `src/pairwise_rank/providers/`. One file per provider.
4. Use the provider's default sampling behavior. Do not send
   temperature / top_p / penalties / seed / beam unless the
   provider API requires one of them.
5. Request the maximum supported reasoning explicitly, if the
   API exposes one. Document the field name and value in the
   connector's module docstring.
6. Map the provider response into the library's canonical
   `Judgment` type: `(verdict, reasoning, provider, model,
   reasoning_effort)`. If the provider's structured-output
   schema differs from the canonical schema, adapt inside the
   connector. Do not weaken the canonical internal contract to
   mirror a provider API.
7. Preserve raw provider metadata in the `Judgment.raw` field
   for debugging and audit. Do not log the raw response to
   stdout by default.
8. Read authentication from the environment. Document the env
   var name in the connector docstring. Never hard-code
   credentials. Never commit secrets.
9. Add contract tests under `tests/test_providers/`. Mock the
   HTTP layer. Test at least: LEFT/TIE/RIGHT mapping, malformed
   response fails loudly, no temperature / no top_p, reasoning
   setting is present, auth header is sent.
10. Do not modify statistical code. Adding a new provider
    should not require editing `btd.py` or `protocol.py`.

If structured output / tool-use support differs between
providers, the adapter lives inside the connector. The
canonical internal judgment contract stays fixed.

---

## 11. Operational rules (preserved)

These rules were derived from concrete failures. They are
non-negotiable. They are project-specific conventions that
agents have been observed to violate.

### Never silently edit user-provided values

Specific banned edits:

- Changing a user-specified reasoning value to "save tokens"
- Switching tool choice from one mode to another because one is
  "flakier"
- Modifying function names from what the user said
- Truncating or paraphrasing user-provided instructions
- Removing tools / parameters / descriptions to "simplify"
- Adding tools / parameters / descriptions to "improve"
- Switching image detail from "high" to "low"/"auto"

If the user-provided values are wrong, ask. If they are right,
use them verbatim.

### Show the body before running

For any new experiment or any change to the design, **show
the user the full request body before sending it to the API.**
This is not optional. Even if the body is "obviously the same
as last time."

### Repository hygiene before any commit

Before `git commit`:

1. Run `git status --short` and `git diff --stat` to see what
   is staged. Unstage personal-experiment artifacts.
2. Verify the version in `pyproject.toml` and
   `src/pairwise_rank/__init__.py` is consistent if the change
   is user-facing.
3. Verify the test suite passes.
4. Verify `git log -1` shows the expected commit message.

### Push protocol

In some sandbox environments, GitHub's SSL certificate
verification fails. Use:

```
git -c http.sslVerify=false push
```

This workaround is accepted. Do not pretend the underlying
issue is solved.

### PyPI protocol

This package is published to the canonical PyPI. Use:

```
--index-url https://pypi.org/simple/ --break-system-packages
```

Do not waste time fighting any mirror.

### When in doubt, ask once, briefly

Silent changes are worse than brief questions. If a value is
ambiguous, ask. If the answer is obvious, do it.

The test: would the user be surprised by what I'm about to
do? If yes, ask. If no, do it.

---

This file is the source of truth for repository policy.
`EXPERIMENT_DESIGN.md` holds the design methodology. Both
must be respected.
