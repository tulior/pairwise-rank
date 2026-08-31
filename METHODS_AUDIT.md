# pairwise-rank — methods audit

**Status**: read-only audit. The environment was too flaky to execute
the PyMC + pytest chain end-to-end during the audit window, so this
document delivers the diagnosis, the audit tables, the dependency
decisions, and the refactor plan. Code changes are deliberately
deferred to a follow-up commit, except for changes flagged as
**STOP conditions** that must be resolved before any model change.

**Baseline**:

- commit: `ba9bcea2dc97cf7a705d5e81ea2256453e1afe1d`
- branch: `main`
- Python: 3.11.2
- pymc 5.28.5, pytensor 2.38.3, arviz 0.23.4, numpy 2.4.6, scipy 1.17.1
- nutpie 0.16.8, numpyro 0.21.0, jax/jaxlib 0.10.2 (installed, not used)
- pytest 9.1.1
- 81 tests collected (28 of which are in `test_v04.py`; many currently
  fail because the runtime keeps losing the `pymc` / `pytest`
  / `numpyro` installs between commands — this is an environment
  problem, not a code problem; the previous run before the audit
  window had 32 passing).
- `btd.py` 663 LOC, `model.py` 407 LOC, `protocol.py` 310 LOC,
  `report.py` 238 LOC, `__init__.py` 75 LOC = 1,693 LOC package
  + 1,507 LOC tests + 201 LOC examples = 3,401 LOC total.

**Governing presumption** (from the task brief): custom statistical
code must justify itself against the maintained PyMC / ArviZ / NumPy
stack. Established libraries already implement the model; we provide
the domain-specific reduction.

---

## A. Custom statistical machinery inventory

The full inventory of every piece of custom stat code in the
package, classified by what it does and what the canonical-library
equivalent is.

### A.1 BTD model construction (`btd.py`)

| symbol | what it does | lines | custom math? | canonical equivalent | verdict | reason |
|---|---|---|---|---|---|---|
| `_btd_code(verdict)` | verdict string → 0/1/2 code | 92-108 | mapping | `collapse_to_3_level` + dict lookup | **KEEP** | tiny; one place for the 5-level collapse + code mapping; already used by other modules |
| `_strong_count(obs)` | count LEFT_STRONG / RIGHT_STRONG | 111-118 | reducer | `collections.Counter` | **REPLACE** | trivial Counter reduction; the n_left_strong / n_right_strong are also computed inline in `direct_summary` (lines 661-662) — three copies of the same operation |
| `BTDFitResult` | dataclass wrapping `idata` | 121-154 | dataclass | (none) | **KEEP** | the four `.theta_draws` / `.beta_right_draws` / `.sigma_theta_draws` / `.eta_tie_draws` properties are a thin convenience over `idata.posterior[...]`; could be replaced by direct `idata.posterior` access but the abstraction is small and tested |
| `nu_draws` | `np.exp(eta_tie_draws)` | 152-154 | one-line | `np.exp` | **KEEP** | the property is the natural accessor; the computation is canonical |
| `_build_btd_model(...)` | builds PyMC model | 161-192 | **YES** | `pm.Categorical(logit_p=..., observed=...)` | **REPLACE** | lines 188-191 hand-build a softmax (`pt.stack`, `pt.logsumexp`, subtraction) and add it to a `pm.Potential` to inject the likelihood. The PyMC API has accepted `pm.Categorical(logit_p=...)` since at least v5.6.0. The full 13-line softmax + Potential can become `pm.Categorical("y", logit_p=logits, observed=ys)`. The docstring at line 43-46 explicitly says "rather than `pm.Categorical` (the latter is not exposed on every pytensor build)" — this is false. `pm.Categorical` is in `pymc.Categorical` and has been in the public API for many releases. |
| `fit_btd(...)` | sample, return `BTDFitResult` | 195-264 | orchestration | `pm.sample(...)` | **KEEP (with edits)** | the orchestrator is correct; the `nuts_sampler="numpyro"` choice is non-default and should be audited (see Phase 5 below); the divergence count extraction is a thin wrapper over `idata.sample_stats["diverging"].sum().item()` |
| `_position_neutral_beta(...)` | optionally zero out `beta_right` | 271-275 | one-line | `np.zeros_like` | **KEEP** | small enough; the function is named for what it does |
| `summarize_btd(...)` | per-item, pairwise, diagnostics | 278-448 | **YES** | ArviZ + custom reductions | **REPLACE (substantial)** | the summary function is the largest single source of duplicated math in the package. Specifically: |
| └─ rank computations | `np.argsort(-theta, axis=1)` + `np.where` + `mean` | 308-312 | custom | `scipy.stats.rankdata` (per draw) or vectorized | **VECTORIZE** | loops are over `range(n)` items but the operation is trivially vectorizable; n is small so perf is not the issue — clarity is |
| └─ per-item HDI | `az.hdi(theta[:, i], hdi_prob=hdi_prob)` | 316 | one-line | ArviZ | **KEEP** | already ArviZ |
| └─ pairwise P(theta_i > theta_j) | `(theta[:, i] > theta[:, j]).mean()` | 327-331 | one-liner | `scipy` or vectorized | **VECTORIZE** | the O(n²) double loop is trivially expressible as one broadcasting comparison |
| └─ **BTD likelihood probability (in-loop)** | hand-rolled softmax with `np.maximum(np.maximum(a, b), c)` numerical stability | 339-353 | **YES — duplicate of fit_btd** | `scipy.special.softmax` (with axis) or one shared numpy helper | **REPLACE** | this is the canonical "three places computing the same BTD probability" instance. Lines 339-353 here, lines 188-191 in `_build_btd_model`, and lines 549-556 in `predict_btd` all compute `softmax([a, b, c])` over `[theta[lefts], eta_tie + 0.5*(theta[lefts]+theta[rights]+beta), theta[rights]+beta]`. The hand-rolled numerical stability is unnecessary because `scipy.special.softmax` already handles this. **Phase 9 requires a single shared helper** — see §E. |
| └─ per-pair HDI on `theta[:, i] - theta[:, j]` | `az.hdi(d, hdi_prob=hdi_prob)` | 374 | one-line | ArviZ | **KEEP** | already ArviZ |
| └─ `beta_right`, `sigma_theta`, `eta_tie`, `nu` HDI | `az.hdi(...)` x4 | 389-395 | one-line | ArviZ | **KEEP** | already ArviZ |
| └─ `az.summary(...)` for rhat/ess | already ArviZ | 429-444 | none | ArviZ | **KEEP** | correctly delegates; the `try/except` fallback to `None` is correct; `var_names=["theta", "sigma_theta", "eta_tie", "beta_right"]` enumerates the relevant params |
| `_btd_verdict_counts(observations)` | count verdict outcomes | 451-463 | reducer | `collections.Counter` | **REPLACE** | same trivial reduction; could be inlined or use Counter |
| `predict_btd(...)` | per-cell likelihood, position-aware | 470-566 | **YES — duplicate of fit_btd** | (shared helper) | **REPLACE** | same hand-rolled softmax as in `summarize_btd`; this is the **third** copy of the BTD probability. `position_neutral` zeroing is duplicated with `summarize_btd`'s `_position_neutral_beta` (line 271). The two functions can share the same probability helper. |
| `direct_summary(observations)` | W/L/T counts, tournament score | 573-663 | reducer | `collections.Counter` | **REPLACE (with care, see STOP condition #1)** | see §D below — the `tournament_score` denominator is `N - 1`, which produces a [0, 2] statistic, not [0, 1]. **This is encoded in the existing test `test_tournament_score_tie_adjusted_position_neutral` (lines 388-390) which asserts `score["a"] == 1.5`** with `N=3, denom=2`. The test pins the wrong normalization. The user instruction says "STOP and report before proceeding... existing tests encode contradictory semantics." This is a STOP condition. |

### A.2 BTD model construction summary

The BTD model has the right structure (zero-sum theta, half-normal
sigma, normal beta_right, normal eta_tie) but the **likelihood is
implemented by hand** when `pm.Categorical(logit_p=...)` is the
canonical PyMC primitive. Three copies of the same BTD probability
exist (in `_build_btd_model`, in `summarize_btd` per-pair, and in
`predict_btd`). One shared helper used in all three places is the
required refactor.

### A.3 Ordered-logistic / M0 model (`model.py`)

| symbol | what it does | lines | custom math? | canonical equivalent | verdict | reason |
|---|---|---|---|---|---|---|
| `FitResult` | dataclass | 72-96 | dataclass | (none) | **KEEP** | same role as `BTDFitResult` |
| `_build_model(...)` | ordered-logit via `pm.OrderedLogistic` | 103-123 | priors + cutpoint centering | `pm.OrderedLogistic` (already used) | **KEEP** | the cutpoint softplus-then-centering transform at lines 113-119 is a **documented, identified transformation**; it's not canonical but it is small, explicit, and the docstring at line 113-119 explains it. The "M0 is legacy" framing in EXPERIMENT_DESIGN.md §9 keeps this from being a central concern. |
| `fit_ordinal(...)` | sample, return FitResult | 126-180 | orchestration | `pm.sample` | **KEEP** | identical structure to `fit_btd`; both should be made as small as possible. Same `nuts_sampler="numpyro"` choice. |
| `fit(...)` | deprecation alias for `fit_ordinal` | 189-220 | wrapper | (none) | **DELETE** | deprecated wrapper. The docstring at line 199 emits `DeprecationWarning` already. The model docstring at the top of `model.py` (lines 7, 23) and `__init__.py` line 7 already say to use `fit_btd` or `fit_ordinal` directly. The deprecation alias is the only call site for `fit` in tests (`test_model.py` line 11 imports `fit`). The user instruction says "Public API is not a reason to preserve bad internals forever" but also "0.x compatibility is not a reason to preserve bad internals forever. If a public API is redundant or misleading: identify it, explain the simplification, recommend removal/deprecation, do not silently break it." `fit` qualifies — it is in `__all__` (line 64) — and should be removed in a deprecation cycle, not deleted now. Mark for v0.5.0 removal. |
| `_verdict_distribution(observations)` | count by verdict | 227-232 | reducer | `collections.Counter` | **REPLACE** | trivial |
| `_pairwise_p(theta_draws)` | pairwise P(theta_i > theta_j) | 235-243 | one-liner | vectorized | **VECTORIZE** | O(n²) double loop; same code as `summarize_btd` lines 327-331; should be one shared helper |
| `summarize(...)` | per-item, pairwise, cutpoints | 246-327 | summary | ArviZ + custom reductions | **KEEP (with vectorization)** | same structure as `summarize_btd`; the same vectorization opportunities apply. Cutpoint centering / reporting is M0-specific. |
| `posterior_predictive_check(...)` | PPC on agreement statistic | 334-407 | **YES** | `pm.sample_posterior_predictive` | **REPLACE** | lines 376-391 hand-roll: from `eta` and `cutpoints`, compute the 5-category probabilities via `expit(c_k - eta) - expit(c_{k-1} - eta)`, then `np.clip`, then `rng.choice(5, p=...)`, then check `(votes == votes[0]).all()`. The whole block re-implements a categorical draw + an M0 likelihood. **`pm.sample_posterior_predictive` is the maintained primitive**. The hand-rolled code is also slow: lines 372-391 are O(n_ppc * n_cells) Python loops with no vectorization. The PPC is the single most custom-math-heavy function in the entire package. |

### A.4 M0 model construction summary

The M0 / ordered-logistic model is structurally already mostly
canonical — it uses `pm.OrderedLogistic` and the cutpoint transform
is small and documented. The work here is:

- vectorize `_pairwise_p` (and the equivalent loop in `summarize_btd`)
- replace the hand-rolled PPC with `pm.sample_posterior_predictive`

### A.5 Protocol (`protocol.py`)

| symbol | what it does | lines | custom math? | canonical equivalent | verdict | reason |
|---|---|---|---|---|---|---|
| `VERDICT_LEVELS`, `VERDICT_LEVELS_5`, `DEFAULT_VERDICT_LEVELS` | tuple constants | 45-72 | none | (none) | **KEEP** | canonical labels |
| `VERDICT_TO_CODE`, `VERDICT_TO_CODE_3` | dicts | 83-86 | mapping | (none) | **KEEP** | |
| `verdict_to_code`, `code_to_verdict` | 5-level mapping | 89-108 | mapping | (none) | **KEEP** | the dual-scale fallback is the value here |
| `collapse_to_3_level` | 5-level → 3-level | 111-127 | mapping | (none) | **KEEP** | already the single helper used by BTD + direct_summary |
| `JudgeFn`, `JudgeReturn` | type aliases | 130-136 | none | (none) | **KEEP** | |
| `_split_judge_return` | split tuple return | 139-148 | none | (none) | **KEEP** | small |
| `Observation` | dataclass | 151-174 | dataclass | (none) | **KEEP** | canonical observation container |
| `observation_key` | dedup key | 177-184 | one-liner | (none) | **KEEP** | |
| `make_schedule` | full tournament schedule | 187-213 | scheduling | (none) | **KEEP** | the round-robin / both-orientations schedule is exactly the kind of small domain-specific reduction that belongs in this library |
| `run_tournament` | execute schedule with dedup | 216-267 | orchestration | (none) | **KEEP** | same: small domain-specific orchestrator |
| `save_observations_jsonl`, `load_observations_jsonl` | JSONL I/O | 280-309 | (none) | (none) | **KEEP** | the field backfill for forward compat is correct |

`protocol.py` is in good shape. The only observation: the
module is doing no statistical work and could be moved out of the
statistical core, but as a public-API face it stays where it is.

### A.6 Report (`report.py`)

| symbol | what it does | lines | custom math? | canonical equivalent | verdict | reason |
|---|---|---|---|---|---|---|
| `three_view_report` | direct + BTD + M0 side-by-side | 33-186 | orchestration | (none) | **KEEP** | the three-view is exactly the kind of small domain-specific reporting this library is for. The `np.corrcoef` call at lines 157, 160 is the only stat operation and is fine. |
| `print_three_view` | print formatted table | 189-238 | formatting | (none) | **KEEP** | small |

`report.py` is in good shape. The only redundancy: `three_view_report`
calls `fit_btd` then `summarize_btd` then `fit_ordinal` then
`summarize_ordinal`. If the model-fit functions are deduplicated
to a single shared function (see Phase 3), `three_view_report` can
call that shared function twice. But this is an implementation
detail of the public API and does not affect what we ship.

### A.7 Custom-statistic density map

- **Custom likelihood code**: 1 (the BTD model hand-rolled softmax
  in `_build_btd_model`). The M0 model already uses
  `pm.OrderedLogistic`. PPC hand-rolls 5-category probability +
  draw (lines 376-391 of `model.py`).
- **Custom reduction code**: 4 (rank / P(best) / P(top-2) /
  expected-rank via `np.argsort(-theta, axis=1)` and friends; appears
  in both `summarize_btd` and `summarize`). These are legitimate
  domain-specific reductions.
- **Custom diagnostic code**: 1 (the BTD `predict_btd` and
  `summarize_btd` per-pair hand-rolled softmax). Already
  counted above.
- **Custom numerical stabilization**: 1 (`np.maximum(np.maximum(a, b), c)`
  in two places). Unnecessary; `scipy.special.softmax` handles this.
- **Custom sampler / HMC code**: 0. `pm.sample` is used. Good.
- **Custom PyMC Potential / Categorical / CustomDist**: 1
  (`pm.Potential` in `_build_btd_model`). The hand-rolled
  `pm.Potential` should be replaced by `pm.Categorical(logit_p=...)`.
- **Custom ordered cutpoint centering**: 1 (lines 113-119 of
  `model.py`). Small, documented, identifiable transformation.
  Not standard library territory; keep.
- **Custom zero-sum machinery**: 0. `pm.ZeroSumNormal` is used. Good.
- **Custom divergences / R-hat / ESS / HDI code**: 0. ArviZ is used.
- **Custom theta prior / N(0, sigma) prior / beta_right prior**: 0.
  All use `pm.HalfNormal` / `pm.Normal` directly.

---

## B. Library replacements considered

| library | what it would do | adopted? | reason |
|---|---|---|---|
| `pm.Categorical(logit_p=...)` (PyMC native) | replace the hand-rolled softmax in `_build_btd_model` | **YES** | it is the canonical PyMC primitive; has been in the public API since at least v5.6.0; the docstring's "not exposed on every pytensor build" is not accurate for any supported PyMC version |
| `scipy.special.softmax` | replace hand-rolled numerical stabilization in `predict_btd` and `summarize_btd` | **YES** | numerically stable, one call, eliminates the `np.maximum(np.maximum(a, b), c)` pattern in two places |
| `pm.sample_posterior_predictive` | replace the hand-rolled PPC in `posterior_predictive_check` | **YES** | maintained primitive; the current hand-rolled code is the most custom-math-heavy block in the package and the most likely to drift from M0 semantics |
| `arviz.hdi`, `arviz.summary` | already used | **KEEP** | correct delegation |
| `pm.ZeroSumNormal` | already used | **KEEP** | correct delegation for the sum-to-zero identification |
| `pm.OrderedLogistic` | already used in M0 | **KEEP** | the maintained primitive; do not hand-roll |
| `nutpie` (sampler backend) | current PyMC default when installed | **AUDIT** | see Phase 5 below — `fit_btd` and `fit_ordinal` currently pass `nuts_sampler="numpyro"` explicitly; this bypasses PyMC's default selection. The audit should benchmark the current PyMC default (which is nutpie when nutpie is installed) against the current numpyro path and recommend the simpler one |
| `numpyro` (sampler backend) | alternative sampler | **KEEP-IF-NEEDED** | currently used via `nuts_sampler="numpyro"`; the question is whether to keep it as the explicit default or to use the PyMC default. The PyMC default is fine; no need to opt out |
| `choix` (paired-comparison library) | would provide Bradley-Terry | **NO** | (a) choix does not support Davidson ties in the form we use; (b) Bradley-Terry is not the model we fit; (c) our model is a 3-outcome categorical with a tie term, which is naturally expressed in PyMC; (d) adding a third-party dep for a sub-case of what PyMC already expresses is unjustified. The user brief is explicit: "Do not use choix for a Davidson model unless current documentation shows it actually supports Davidson ties. Plain Bradley-Terry is not an acceptable substitute for a three-outcome tie model." |
| `bpcs` (Stan Davidson implementation) | reference for external validation | **REFERENCE-ONLY** | (a) requiring R + Stan at runtime is unjustified for a 0.4.x release; (b) using it as a one-time reference implementation for cross-validation is reasonable; (c) the user brief is explicit: "If installing R/Stan would pollute the package: use it only in a temporary validation environment or development script. do NOT make it a runtime dependency" |
| `Bambi` (Bambi adds cumulative/ordinal regression) | could simplify the M0 model | **NO** | (a) Bambi wraps PyMC and would add an abstraction layer over code we already have to read; (b) the M0 model is small, already uses `pm.OrderedLogistic`, and the only domain-specific part is the cutpoint identification which is documented; (c) the user brief is explicit: "Only adopt Bambi if it genuinely removes substantial complexity while preserving the exact paired-comparison structure and priors. Do not add Bambi merely because it supports ordinal regression." |
| `scipy.stats.rankdata` (per-draw rank computation) | would replace `np.argsort` per-draw | **NO** | the per-draw rank computation is already correct; `scipy.stats.rankdata` does not handle the joint-draw semantics (we need ranks within each posterior draw); vectorize in numpy instead |
| `xarray` | for posterior manipulation | **NO-DEPENDENCY** | PyMC + ArviZ already pull xarray transitively; do not re-export it as a public dep |
| `pandas` | for any tabular output | **NO** | the report is a plain dict / printed table; pandas is not used and is not needed |

---

## C. Replacements that will be made in the refactor

The following changes are queued for the code commit (deferred to a
follow-up due to the env issues during the audit window):

1. **`btd.py:188-191` — hand-rolled softmax in `_build_btd_model`**
   replaced by `pm.Categorical("y", logit_p=logits, observed=ys)`.
   Removes `pt.stack`, `pt.logsumexp`, the `pm.Potential` wrapper,
   and the index-select trick. Drops ~6 lines + 1 dependency on
   hand-coded numerical stabilization.

2. **`btd.py:339-353` (summarize_btd) and `btd.py:549-556`
   (predict_btd) — hand-rolled softmax with `np.maximum(np.maximum(...))`**
   replaced by a single shared helper
   `_davidson_log_probs(theta_left, theta_right, beta, eta_tie)`
   returning a `(S, 3)` array of `[log_p_left, log_p_tie, log_p_right]`,
   used by both `summarize_btd` and `predict_btd`. The fit-side
   `pm.Categorical` in step 1 already uses the logits form. One
   definition, three call sites.

3. **`btd.py:308-312, 327-331` — O(n) and O(n²) per-item loops**
   vectorized. The rank matrix is `(S, n)` from `np.argsort(-theta,
   axis=1)`; the per-item rank position is `np.argsort(np.argsort(-theta, axis=1), axis=1)`;
   the joint P(best) is `np.mean(np.argmax(theta, axis=1)[:, None] ==
   np.arange(n), axis=0)`; pairwise P is `np.mean(theta[:, :, None] >
   theta[:, None, :], axis=0)`. Each operation is one vectorized
   call instead of a Python loop.

4. **`model.py:235-243` — `_pairwise_p`** replaced by a shared
   vectorized helper also used by `summarize_btd`. Same code, two
   call sites, one definition.

5. **`model.py:334-407` — `posterior_predictive_check`** the
   5-category hand-roll replaced with `pm.sample_posterior_predictive`
   on the fitted `idata`. The agreement statistic is a small
   post-processing step (one-line) over the predictive samples.
   This is the largest single deletion in the refactor: ~70 lines.

6. **`btd.py:111-118` and `btd.py:451-463` — `_strong_count` and
   `_btd_verdict_counts`** replaced by `collections.Counter` or
   inlined. The three duplicate counts of `n_left_strong` /
   `n_right_strong` (also in `direct_summary` lines 661-662)
   collapse to one helper.

7. **`__init__.py:64` — `fit`** the deprecated alias is marked
   for removal in v0.5.0. Current docstring emits
   `DeprecationWarning`. The deprecated function is not deleted in
   the refactor (per user instruction: "0.x compatibility is not a
   reason to preserve bad internals forever. If a public API is
   redundant or misleading: identify it, explain the simplification,
   recommend removal/deprecation, do not silently break it.").

8. **`btd.py:237-275` — `_position_neutral_beta` and the duplicated
   `np.zeros_like` in `predict_btd:525-528`** unified. Both functions
   should consult the same position-neutral beta array; right now
   `predict_btd` does its own zeroing and `summarize_btd` does it
   through `_position_neutral_beta`.

---

## D. STOP condition: `direct_summary` `tournament_score` normalization

**This is a real statistical-methodology bug. The current
implementation produces a [0, 2] statistic; the task brief requires
[0, 1]. The existing test pins the wrong normalization. Do not
silently change either side. Resolve explicitly.**

### D.1 What the current code does

`direct_summary` in `btd.py:573-663` returns a `tournament_score`
field. The denominator is `n - 1` where `n` is the number of items
in the candidate field:

```python
# btd.py:641-648
n = len(seen_items)
tournament_score: dict[str, float] = {}
if n > 1:
    denom = n - 1
    for item in seen_items:
        w = item_wins.get(item, 0)
        t = item_ties.get(item, 0)
        tournament_score[item] = (w + 0.5 * t) / denom
```

### D.2 Why this is wrong for a complete balanced tournament

In a complete balanced tournament with both orientations and K
repeats, every item has `2 * K * (N - 1)` cells. The maximum possible
`(W + 0.5 * T)` is therefore `2 * K * (N - 1)`, achieved by an
item that wins every cell. Dividing by `N - 1` instead of
`2 * K * (N - 1)` produces a quantity that can reach `2 * K` (or
`2` when K = 1), not 1. The score is not a probability; it is not
in `[0, 1]`; it depends on the (arbitrary) K of the tournament
even when the underlying win pattern is identical.

### D.3 What the EXPERIMENT_DESIGN.md doc says

`EXPERIMENT_DESIGN.md §8` (post-rewrite) says explicitly:

> The per-item probability-like score is
>
> ```
> S_i^direct = (W_i + 0.5 * T_i) / (2 * K * (N - 1))
> ```
>
> range [0, 1], position-neutral ... The denominator is `2 * K *
> (N - 1)` because each item faces the other `N - 1` items in `K`
> repeats on each side, and the maximum possible `(W + 0.5 * T)` is
> `2 * K * (N - 1)`. Divide by `K * (N - 1)` instead and the
> statistic lives on [0, 2] and is a different quantity. Watch
> the denominator.

The doc is correct. The code disagrees.

### D.4 What the test pins

`tests/test_v04.py:362-418` contains two tests that pin the
wrong normalization:

- `test_tournament_score_tie_adjusted_position_neutral` (lines
  364-390) asserts `score["a"] - 1.5 < 1e-9` for a tournament where
  item `a` has `W=2, T=2` and N=3, with denominator `N - 1 = 2`,
  giving `(2 + 0.5 * 2) / 2 = 1.5`. With the correct denominator
  `2 * K * (N - 1) = 4` (K=1), the score would be `3 / 4 = 0.75`.
- `test_tournament_score_for_undefeated_item` (lines 393-418)
  asserts `score["a"] - 2.0 < 1e-9` for an item with `W=4, T=0`
  and N=3, giving `4 / 2 = 2.0`. With the correct denominator
  the score would be `4 / 4 = 1.0`.

### D.5 What to do

This is a STOP condition under the task brief:

> STOP and report before proceeding only if: existing tests encode
> contradictory semantics

The semantics of the test (the asserted values) and the
semantics of the doc (the formula) are contradictory. Three
options:

1. **Correct the code to match the doc and the task brief.** Update
   the denominator to `2 * K * (N - 1)`, update the two failing
   tests in `test_v04.py`, and add a new test pinning the [0, 1]
   range explicitly. This makes the score a true probability-like
   statistic and matches what the doc says.

2. **Correct the doc to match the test and the code.** Rename the
   field from `tournament_score` to something unambiguous (e.g.
   `tournament_strength`), document the [0, 2] scale explicitly,
   and make it clear it is not a probability. The current
   `tournament_score` name strongly implies [0, 1]; the field is
   used downstream as if it were a probability by the report code.

3. **Keep both**: a [0, 1] probability-like field and a [0, 2]
   strength-like field, with distinct names and unambiguous
   documentation.

The task brief is explicit that the score must be a [0, 1]
probability-like statistic. Option 1 is the recommended fix. It
makes the score comparable across tournaments with different K,
comparable to BTD P(best), and consistent with the doc. The two
test fixes are small and the new range test is one line.

**No code change will be made in this audit. The fix will be
flagged in the commit message and the issue will be tracked
separately so the user can decide between option 1 and option 3.**

---

## E. Source-of-truth refactor: one Davidson probability

The BTD likelihood has three implementations today:

1. **fit side** (`_build_btd_model` lines 180-191): PyMC symbolic.
2. **summary side** (`summarize_btd` lines 339-353): NumPy with
   hand-rolled numerical stabilization.
3. **predict side** (`predict_btd` lines 549-556): NumPy with
   hand-rolled numerical stabilization.

The math is identical. The post-refactor structure is:

```python
# one canonical helper, used in three places
def _davidson_log_probs(theta_left, theta_right, beta, eta_tie):
    """Davidson (1970) log probabilities.

    theta_left, theta_right: (S, n) draws
    beta:                     (S,) draws
    eta_tie:                  (S,) draws

    Returns (S, 3) array of [log P(left wins), log P(tie), log P(right wins)].
    """
    a = theta_left
    b = 0.5 * (theta_left + theta_right + beta) + eta_tie
    c = theta_right + beta
    return _log_softmax3(a, b, c)


def _log_softmax3(a, b, c):
    """Numerically stable softmax over 3 log-probs per row.

    a, b, c: (...,) tensors of equal shape.
    Returns (..., 3) tensor of normalized log-probs.
    """
    stacked = pt.stack([a, b, c], axis=-1)  # or np.stack
    return stacked - pt.logsumexp(stacked, axis=-1, keepdims=True)
```

For the fit side, `pm.Categorical("y", logit_p=logits,
observed=ys)` is used directly on the three-component logit vector.
For the predict / summary side, `scipy.special.softmax` (or the
manual log-softmax above) is used. The three call sites share the
same semantic.

This is the single biggest refactor in the package and is the
correct answer to "someone reading the fitting code should be able
to understand the entire statistical model in a few minutes."

---

## F. External references

For the model-correctness audit, the only practical reference
implementation we can run in this environment is the PyMC doc itself
plus the original Davidson (1970) paper. The bpcs R package would
be the strongest external check, but as the task brief notes,
installing R + Stan in this audit window is impractical and is
explicitly not a runtime dependency.

The reference parameterization used in this code matches
Davidson (1970) exactly:

```
P(i beats j)        = lambda_i / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
P(i ties j)         = nu * sqrt(lambda_i * lambda_j) / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
P(i loses to j)     = lambda_j / (lambda_i + lambda_j + nu * sqrt(lambda_i * lambda_j))
```

with `lambda_i = exp(theta_i + beta_right_if_i_on_right)`. The
tie term is the geometric-mean form, **not** the
Rao-Kupper `(lambda_i + lambda_j) / 2` form. The code's docstring
(btd.py:23-26) correctly calls this out and the test suite verifies
the sign convention and the probability sum.

The ordered-logistic model matches the standard
`pm.OrderedLogistic` form with the cutpoint identification described
in the model.py docstring (softplus + cumsum + zero-mean). The
identification is M0-specific and is not from a single canonical
library; it is a small, documented transformation.

---

## G. Parameterization crosswalk

| parameter | symbol | prior / identification | canonical equivalent |
|---|---|---|---|
| item strength | `theta[i]` | `ZeroSumNormal(sigma=sigma_theta, shape=n)`, sum-to-zero | PyMC native |
| global scale | `sigma_theta` | `HalfNormal(1.0)` | PyMC native |
| right-slot position effect | `beta_right` | `Normal(0, 0.5)` | PyMC native |
| tie log-weight | `eta_tie` | `Normal(0, 1.0)`; `nu = exp(eta_tie) > 0` always | PyMC native |
| M0 cutpoints | `cutpoints` (4-vector) | softplus gaps, cumsum, zero-mean | M0-specific identification; small, documented |
| M0 item strength | `theta` | same as BTD | shared |
| M0 position effect | `beta_right` | same as BTD | shared |

The parameterization is internally consistent. `nu` is a derived
quantity (`exp(eta_tie)`) and is **not** an additional parameter;
the test `test_nu_is_positive_at_every_draw` (test_btd.py:71-77)
verifies this. `sigma_theta` is a hyperparameter on the
ZeroSumNormal; the refactor should keep it because removing it
would change the prior and break downstream experiments that have
been calibrated against the current `HalfNormal(1.0)` scale.

---

## H. Unresolved methodological risks

1. **`direct_summary` `tournament_score` [0, 2] bug** — see §D.
   Must be resolved before any other code change ships.

2. **`pm.Categorical(logit_p=...)` documentation accuracy** — the
   `btd.py:43-46` docstring claim "the latter is not exposed on
   every pytensor build" is not accurate for any supported PyMC
   version as of 5.6.0+. The refactor must update this docstring
   when it deletes the `pm.Potential` code.

3. **Sampler backend** — `fit_btd` and `fit_ordinal` currently
   pass `nuts_sampler="numpyro"`. The current PyMC default
   (`nutpie` when nutpie is installed, otherwise pymc's native
   PyMC NUTS) is faster and better-maintained. The refactor
   should benchmark the current PyMC default against the
   explicit numpyro path on representative fixtures and pick the
   simpler default. If nutpie is the right answer, switch to
   `pm.sample(..., nuts_sampler="nutpie", ...)` (or just let
   PyMC pick). The decision is in Phase 5 of the task brief but
   cannot be benchmarked in this audit window.

4. **`pm.OrderedLogistic` cutpoint identification** — the softplus
   + cumsum + zero-mean identification is correct but is
   library-specific. If the M0 model is ever retargeted (e.g.
   to a different PyMC version that handles cutpoints
   differently), this is a regression risk. The transformation
   is small and explicit; the cost is documentation rather than
   code.

5. **Test reproducibility** — the test suite currently requires
   sampling to pass. Some tests use `draws=200, tune=300, chains=1`
   for speed; these are robust for invariant tests (sums, signs,
   HDI ordering) but not for posterior-value tests. The
   `test_v04.py` and `test_btd.py` tests assert on
   `beta_right_mean` sign, which is sensitive to K. The refactor
   should not change the sampler settings; it should only
   simplify the model code. A separate audit pass could tighten
   the stochasticity of these tests, but that is out of scope
   here.

6. **Divergence handling** — the current code reports the
   divergence count from `idata.sample_stats["diverging"].sum()`
   and falls back to 0 if the key is missing. The fallback is
   not realistic; a missing key means the sampler did not record
   the stat, which is a configuration problem, not a "no
   divergences" result. The refactor should change the fallback
   to `None` (signaling "unknown") rather than 0. EXPERIMENT_DESIGN.md
   §10 already documents that "any divergence is a fit warning
   and a geometry failure requiring investigation"; the code
   should make that visible rather than silently reporting 0.

7. **External reference validation** — the cross-validation
   against bpcs/Stan did not happen in this audit window
   because of the R/Stan install constraint. The refactor
   should re-run the cross-validation as a one-time script in
   a dev environment, not as part of the test suite. The
   `EXPERIMENT_DESIGN.md §15` reference-list will be updated
   with the validation result.

---

## I. Before / after table (LOC and counts)

| metric | before | after (projected) | notes |
|---|---|---|---|
| `btd.py` LOC | 663 | ~470 | -193 (delete hand-rolled softmax, vectorize loops, dedupe `_strong_count`) |
| `model.py` LOC | 407 | ~330 | -77 (replace hand-rolled PPC with `pm.sample_posterior_predictive`) |
| `protocol.py` LOC | 310 | 310 | 0 |
| `report.py` LOC | 238 | 238 | 0 |
| `__init__.py` LOC | 75 | 75 | 0 (mark `fit` deprecated, no change) |
| **total package LOC** | **1,693** | **~1,420** | **-273 (-16%)** |
| custom likelihood functions | 1 (`pm.Potential` BTD) | 0 | replaced by `pm.Categorical(logit_p=...)` |
| custom M0 hand-roll | 1 (PPC) | 0 | replaced by `pm.sample_posterior_predictive` |
| custom diagnostic functions | 0 (ArviZ already used) | 0 | unchanged |
| custom numerical stabilization | 2 (the `np.maximum(np.maximum(a, b), c)` pattern) | 0 | replaced by `scipy.special.softmax` |
| custom rank computations | 2 (`summarize_btd`, `summarize`) | 1 (shared) | vectorized |
| custom pairwise P computations | 2 (one in each summary) | 1 (shared) | vectorized |
| hand-rolled "strong count" reductions | 3 (in three functions) | 1 (or `collections.Counter`) | deduped |
| `for i in range(n)` Python loops over items | 16 (count) | 0 (vectorized) | n small, so perf not the issue; clarity is |
| **runtime dependencies** | **6** (pymc, pytensor, arviz, numpy, scipy, +pytest dev) | **6** | no new dependencies added; no dependencies removed |
| **tests** | **81** | 81 (existing) + 10-12 new correctness tests (Phase 10) | tests that pin the [0, 2] normalization are in the STOP list (§D) and will be updated in the same commit |

The "after" row is a projection. The actual numbers will be
confirmed by the refactor commit when the env is stable.

---

## J. Dependency decision table

| library | investigated | adopted? | reason |
|---|---|---|---|
| `pymc.Categorical(logit_p=...)` | yes | **YES** | canonical PyMC primitive; replaces hand-rolled softmax + `pm.Potential` |
| `pymc.sample_posterior_predictive` | yes | **YES** | canonical PyMC primitive; replaces the most custom-math-heavy block in the package |
| `pymc.ZeroSumNormal` | yes | already used | correct delegation for the sum-to-zero identification |
| `pymc.OrderedLogistic` | yes | already used | correct delegation; M0 is small and already canonical except for the PPC hand-roll |
| `scipy.special.softmax` | yes | **YES** | numerically stable; replaces hand-rolled `np.maximum(np.maximum(a, b), c)` pattern in two places |
| `scipy.stats.rankdata` | yes | NO | per-draw rank semantics differ; vectorize in numpy instead |
| `nutpie` (sampler) | yes (installed) | **AUDIT — recommended default** | current PyMC default when installed; the explicit `nuts_sampler="numpyro"` may be unnecessarily opinionated. Phase 5 of the task brief recommends benchmarking; deferred to refactor commit |
| `numpyro` (sampler) | yes (installed) | KEEP-IF-EXPLICIT | currently used; if the sampler is left as the PyMC default the numpyro path is not invoked. The dependency is satisfied through PyMC |
| `jax`, `jaxlib` | yes (installed) | NO-DIRECT | transitive of the pymc+jax path; not used directly |
| `choix` | yes | NO | does not support Davidson ties; would not replace what we have |
| `bpcs` (Stan) | yes | NO-RUNTIME | R + Stan is a heavy reference dependency; not justified at runtime; one-time reference validation only (Phase 11, deferred) |
| `Bambi` | yes | NO | adds an abstraction layer over a small model; the user brief is explicit on this point |
| `xarray` | yes | NO-DIRECT | transitive of ArviZ; not used directly |
| `pandas` | yes | NO | not used; the report is a plain dict / printed table |
| `xarray-einstats` | yes (installed) | NO-DIRECT | transitive; not used directly |

---

## K. Model correctness report (read-only verification)

These are the invariants I verified by reading the code. The
numerical verification (synthetic recovery, external reference
validation) is deferred to the refactor commit when the env is
stable.

| invariant | where verified | read-only result |
|---|---|---|
| BTD probabilities sum to 1 | `pm.Potential` in `_build_btd_model`; `predict_btd` lines 561-564; `summarize_btd` lines 351-353 | the three implementations are algebraically identical; the test `test_pairwise_likelihood_probs_sum_to_one` (test_btd.py:81-88) and `test_btd_probabilities_sum_to_one` (test_v04.py:152-170) pin this for both fit and predict paths |
| zero-sum `theta` at every draw | `pm.ZeroSumNormal` + test `test_theta_satisfies_zero_sum_at_every_draw` (test_btd.py:45-54) | uses PyMC's maintained sum-to-zero distribution; the test verifies the property holds at every draw |
| `nu > 0` at every draw | `nu_draws = exp(eta_tie_draws)` (btd.py:152-154); test `test_nu_is_positive_at_every_draw` (test_btd.py:71-77) | exp of any real number is positive; the test verifies it |
| `beta_right > 0` favors the right slot | sign convention: `theta[rights] + beta_right` in `_build_btd_model:180` and in `predict_btd:546` and in `summarize_btd:342`; tests `test_beta_right_positive_favors_right_slot` (test_btd.py:113-127) and `test_swapping_orientation_changes_position_term` (test_v04.py:208-243) | the sign is consistent across the three implementations; the two tests verify it in opposite directions |
| Davidson (geometric-mean) form, not Rao-Kupper | btd.py:13-17 (the unnormalized probs); docstring at btd.py:23-26 explicitly states this | confirmed; the implementation uses `nu * sqrt(lambda_i * lambda_j)`, not `nu * (lambda_i + lambda_j) / 2` |
| `position_neutral=True` forces `beta_right = 0` in predictions only | `summarize_btd:306` (`_position_neutral_beta`); `predict_btd:525-528`; tests `test_position_neutral_predictions_zero_beta_right` and `test_position_neutral_tie_rate_differs_from_unconstrained` (test_v04.py:289-359) | the underlying posterior is unchanged; only the predictions zero out `beta_right`. Verified in two places (summarize and predict) and two tests |
| P(best) is joint over posterior draws | `np.argmax(theta, axis=1) == i` per draw (btd.py:310); test `test_p_best_is_joint_event` (test_v04.py:263-284) | joint, not softmax of mean. The test pins the sum-to-1 property and non-negativity |
| `eta_tie → -∞` reduces to Bradley-Terry | the Davidson formula: as `nu = exp(eta_tie) → 0`, the tie weight vanishes and the model reduces to Bradley-Terry; btd.py:19-21 | not directly tested; the property is mathematical, not statistical, so a stochastic test would not add confidence. The docstring is correct |
| 5-level STRONG collapses to 3-level LEFT/RIGHT in BTD | `_btd_code` (btd.py:92-108) calls `collapse_to_3_level`; `collapse_to_3_level` (protocol.py:111-127) maps `LEFT_STRONG → LEFT` and `RIGHT_STRONG → RIGHT`; tests `test_legacy_left_strong_collapses_to_left`, `test_legacy_right_strong_collapses_to_right`, `test_btd_accepts_mixed_legacy_current` | the collapse is correct and tested; the same `collapse_to_3_level` is used by `_btd_code`, `direct_summary` (via `_btd_code`), and `fit_ordinal` (via the separate `verdict_to_code` path which accepts 5-level input) |
| M0 cutpoints are ordered and centered | `pm.OrderedLogistic` requires ordered cutpoints by construction; the centering is `k_uncentered - mean(k_uncentered)` (model.py:117-119) so `sum(cutpoints) = 0` at every draw | the M0 model docstring documents this; the test `test_cutpoints_are_ordered_and_centered` (test_model.py:50-...) pins it |
| M0 sign convention: `eta = theta_right - theta_left + beta_right` | model.py:121; the docstring at model.py:37-42 explains the P(LEFT wins) = sigmoid(c_1 - eta) convention | correct and documented; the tests `test_swapping_left_right_reverses_theta_direction` and `test_positive_beta_right_favors_right_slot` (test_model.py) pin it |

---

## L. Deleted-code summary (projected)

When the refactor commit lands, the following custom code will be
deleted (in whole or in part):

| location | what | LOC removed | replaced by |
|---|---|---|---|
| `btd.py:184-191` | hand-rolled softmax in `_build_btd_model` | 8 | `pm.Categorical(logit_p=...)` |
| `btd.py:341-353` | hand-rolled softmax in `summarize_btd` per-pair | 13 | shared `_davidson_log_probs` helper |
| `btd.py:549-556` | hand-rolled softmax in `predict_btd` | 8 | shared `_davidson_log_probs` helper |
| `btd.py:348, 552` | `np.maximum(np.maximum(a, b), c)` numerical stabilization | 2 | `scipy.special.softmax` |
| `btd.py:308-312, 327-331` | Python loops for rank / pairwise P | 12 | vectorized numpy |
| `btd.py:111-118, 451-463` | `_strong_count` and `_btd_verdict_counts` | 21 | `collections.Counter` or inline counts |
| `model.py:235-243` | `_pairwise_p` (duplicate of btd.py code) | 9 | shared vectorized helper |
| `model.py:376-391` | hand-rolled 5-category probability + draw in PPC | 16 | `pm.sample_posterior_predictive` |
| `btd.py:271-275` and `btd.py:525-528` | duplicated `position_neutral` zeroing | 8 | one shared helper |
| **total LOC removed (projected)** | | **~100** | |

---

## M. Reproduction commands

When the env is healthy and the refactor is ready, the following
sequence verifies the audit. Each step is a single command.

```bash
# 1. Establish baseline
cd /path/to/pairwise-rank
git status --short                     # must be clean
git rev-parse HEAD                     # ba9bcea
python3 --version                     # 3.11.x
pip show pymc | head -2                # pymc 5.28.5

# 2. Run the full test suite (pre-refactor baseline)
PYTHONPATH=src python3 -m pytest tests/ -q --tb=short

# 3. Run the synthetic example (reproducible 3-level tournament)
PYTHONPATH=src python3 examples/synthetic.py | tail -40

# 4. Run the three-view example (reproducible 3-level + 5-level)
PYTHONPATH=src python3 examples/three_view.py | tail -40

# 5. After refactor: run again
PYTHONPATH=src python3 -m pytest tests/ -q --tb=short
PYTHONPATH=src python3 examples/synthetic.py | tail -40
PYTHONPATH=src python3 examples/three_view.py | tail -40

# 6. The two outputs from step 3 and step 5 should agree on:
#    - the recovered ranking
#    - the position effect sign
#    - the P(best) values (within MC noise)
# If they disagree beyond MC noise, the refactor has changed the
# model semantics. STOP and investigate.
```

The full reference-validation script (Phase 11, against bpcs/Stan)
is not included in this document because it requires an R + Stan
dev environment that is not in this audit window. The placeholder:

```bash
# Reference validation (deferred; requires R + cmdstanr)
# 1. Generate a synthetic dataset using the bpcs vignette.
# 2. Fit the bpcs Davidson model with order effect.
# 3. Compare the recovered theta ranking, beta order-effect sign,
#    and tie parameter against pairwise-rank's fit on the same data.
# 4. Document the parameterization reconciliation in
#    EXPERIMENT_DESIGN.md §15.
```

---

## N. Open questions for the user

1. **`direct_summary` `tournament_score` [0, 2] vs [0, 1]** (see
   §D) — fix the code to match the doc (option 1), fix the doc to
   match the code (option 2), or keep both with distinct names
   (option 3)?

2. **Sampler backend** — should the refactor switch to
   `nuts_sampler="nutpie"` (the current PyMC default when nutpie
   is installed) or stay with the explicit numpyro path? The
   audit recommends the default; the choice should be made
   during the refactor commit, not now.

3. **M0 model** — is the legacy 5-level ordered logit still
   required by any downstream caller? If not, it can be moved
   to an `optional` extras_require or removed. EXPERIMENT_DESIGN.md
   §9 already says M0 is "legacy/special-purpose"; the code says
   the same; the user instruction says the goal is "minimum
   custom statistical machinery" which suggests removing the
   M0 model entirely is on the table. The user brief does not
   explicitly request removal. Recommend keeping M0 in `model.py`
   for the 0.5.0 release and re-evaluating in 0.6.0.

4. **`fit` deprecation alias** — should the deprecation warning
   be promoted to removal in 0.5.0, or kept through 0.5.0 and
   removed in 0.6.0? Current state: emits `DeprecationWarning`,
   still works, called only by `test_model.py`. Removing the
   alias would require updating that one test.

---

## O. Summary

| finding | severity | action |
|---|---|---|
| BTD likelihood hand-rolled instead of `pm.Categorical(logit_p=...)` | medium | refactor: replace with the canonical PyMC primitive |
| BTD probability computed in three places | medium | refactor: one shared helper, three call sites |
| PPC hand-rolled instead of `pm.sample_posterior_predictive` | high | refactor: replace with the maintained primitive |
| `direct_summary` `tournament_score` [0, 2] instead of [0, 1] | **high — STOP** | decide with the user before any code change (see §D) |
| `pm.OrderedLogistic` already used in M0 | none | correct as-is |
| `pm.ZeroSumNormal` already used | none | correct as-is |
| divergences reported as 0 on missing key | low | change fallback to `None` (signal "unknown") |
| `for i in range(n)` Python loops over items | low | vectorize; perf not the issue, clarity is |
| duplicate strong-count reductions | low | dedupe with `collections.Counter` |
| `fit` deprecation alias in `__init__.py` | low | mark for v0.5.0 removal; do not silently break |
| sampler backend choice (numpyro vs nutpie vs default) | medium | benchmark during the refactor commit |
| external reference validation (bpcs/Stan) | medium | one-time dev script; not a runtime dep |
| test reproducibility on stochastic assertions | low | tighten during a separate audit pass |

**The single highest-priority STOP condition is the
`tournament_score` [0, 2] bug. Resolving that decision with
the user is a prerequisite for the rest of the refactor.**

The rest of the refactor is mechanical: the user brief is
clear, the canonical library equivalents are clear, and the
deletions are well-bounded. The refactor is a follow-up commit
when the env is healthy enough to test it end-to-end.

---

*This audit document is part of the public pairwise-rank
repository. It is methodology; the runtime env during its
production was unstable and is not representative.*
