# Context handoff

## 1. Mission

Aggressive audit and simplification of `pairwise-rank` (a small
Python library for Bayesian ordinal paired-comparison ranking). The
goal is to make the codebase smaller, more boring, and easier to
audit by removing bespoke statistical machinery wherever PyMC,
ArviZ, scipy, or other maintained libraries already supply a
canonical implementation. Davidson / Bradley-Terry-Davidson
semantics must remain exact, the 3-level LEFT/TIE/RIGHT protocol
must remain the default, and direct counterbalanced evidence
must remain primary for small matched head-to-heads. The
audit-and-simplify pass is *not* a feature expansion and *not* a
generalization. M0 ordered logistic is legacy/special-purpose and
should be allowed to stay minimal.

## 2. Repository state

- **repo path**: `/run/csi/mount-root/nas/eab0d61a99b6696edb3d2aff87b585e8/pairwise-rank`
- **branch**: `main`
- **HEAD commit**: `ba9bcea2dc97cf7a705d5e81ea2256453e1afe1d` — "Rewrite experimental design methodology"
- **git status (compact)**:
  ```
  ?? METHODS_AUDIT.md
  !! .pytest_cache/  build/  src/pairwise_rank.egg-info/  __pycache__/
  ```
  All `!!` entries are gitignored. The only working-tree change is
  the untracked `METHODS_AUDIT.md` (the audit document delivered by
  this session; **not committed**).
- **staged**: nothing
- **unstaged diff to tracked files**: nothing
- **untracked**:
  - `METHODS_AUDIT.md` (48,039 bytes, 720 lines) — the audit
    artifact. **This file is the only session output. It will be
    lost if the working tree is discarded.**
- **safe to discard**: build/, src/pairwise_rank.egg-info/,
  .pytest_cache/, __pycache__/ (all gitignored, generated locally)
- **would be lost if working tree is discarded**:
  - `METHODS_AUDIT.md` — the only non-committed output of this
    session. Commit it before discarding the tree.
- **recent relevant commits**:
  ```
  ba9bcea Rewrite experimental design methodology
  64b341e EXPERIMENT_DESIGN.md §18: coarse-to-fine with sibling-incumbent retention
  4f61102 Document matched-ablation methodology
  35d9504 v0.4.2: Add AGENTS.md and tool-metadata-as-rubric design lesson
  bdb3ec6 Add per-cell BTD predictions and sampler diagnostics
  338da75 Document experiment design principles
  2318939 Make Davidson ranking the default
  bf86796 Demote 5-level ordinal model; BTD + 3-level protocol is the new default
  ```

## 3. Current methodology contract

Semantics that **must not silently change**:

- Default verdicts: `LEFT / TIE / RIGHT` (3-level).
- Legacy 5-level `LEFT_STRONG` / `RIGHT_STRONG` collapse to ordinary
  `LEFT` / `RIGHT` on ingest for BTD and `direct_summary`.
- Direct counterbalanced tallies are primary for small matched
  head-to-heads; the BTD / PyMC global model is a cross-check.
- Davidson / Bradley-Terry-Davidson is the normal global model for
  multi-candidate LEFT/TIE/RIGHT tournaments.
- Tie term is the **Davidson geometric-mean form**,
  `nu * sqrt(lambda_i * lambda_j)`, not Rao-Kupper.
- `beta_right` models the right-slot / order effect.
- `beta_right > 0` means right-slot advantage.
- Position-neutral prediction uses `beta_right = 0` (predictions
  only; the underlying posterior is unchanged).
- `theta` is sum-to-zero (`pm.ZeroSumNormal`).
- `theta` is field-relative; **theta magnitudes from different
  fits are not comparable**.
- `P(best)` is a joint posterior event: the fraction of joint
  posterior draws in which the candidate is the argmax-θ item.
  It is not a softmax of θ means.
- M0 ordered logistic is legacy / special-purpose for genuinely
  ordinal intensity (5-level data). New code uses BTD.
- Divergences are geometry failures requiring investigation.
  There is no "acceptable" divergence fraction. R-hat ≈ 1 is
  necessary, not sufficient. ESS is a Monte Carlo efficiency
  diagnostic, not a coverage measure.
- Reasoning traces are post-hoc audit metadata, not model inputs
  and not guaranteed causal explanations.
- Direct-vs-global strain is not automatically a cycle.
- A cycle requires at least 3 candidates.
- `beta_right` HDI including zero is "not enough data to detect a
  bias", not "no bias". An HDI excluding zero does not imply the
  ranking is trustworthy if the underlying fit is broken.
- Tool metadata (function name, function description, parameter
  names/descriptions/enum, `tool_choice`, message structure,
  reasoning setting, model/provider config, rendering config) is
  part of the experimental instrument. Changing any of these is
  a prompt change.

**Where repo and methodology disagree**: see §9. The
`direct_summary.tournament_score` normalization is the only known
case.

## 4. Public API and important files

### Public API (in `src/pairwise_rank/__init__.py`)

| symbol | module | role |
|---|---|---|
| `VERDICT_LEVELS`, `VERDICT_LEVELS_5`, `DEFAULT_VERDICT_LEVELS` | `protocol` | 3-level / 5-level label tuples |
| `Verdict` | `protocol` | alias for `str` |
| `verdict_to_code`, `code_to_verdict` | `protocol` | 5-level mapping with 3-level fallback |
| `collapse_to_3_level` | `protocol` | 5→3 collapse (the single helper) |
| `JudgeFn`, `JudgeReturn` | `protocol` | type aliases for the judge callable |
| `Observation` | `protocol` | dataclass: `a, b, left, right, repeat, verdict, reasoning` |
| `observation_key` | `protocol` | dedup key (excludes reasoning) |
| `make_schedule`, `run_tournament` | `protocol` | round-robin / both-orientations scheduler with dedup |
| `save_observations_jsonl`, `load_observations_jsonl` | `protocol` | JSONL I/O with forward-compat field backfill |
| `fit_btd` | `btd` | **default probabilistic model** (3-level BTD) |
| `summarize_btd` | `btd` | per-item, pairwise, diagnostics, position_neutral |
| `predict_btd` | `btd` | per-cell orientation-aware BTD likelihood |
| `direct_summary` | `btd` | baseline W/L/T + tournament score (no model) |
| `BTDFitResult` | `btd` | dataclass wrapping `idata`, exposing theta/beta/sigma/eta_tie/nu draws |
| `fit_ordinal` | `model` | **legacy / special-purpose** 5-level ordered logit |
| `summarize` | `model` | per-item / pairwise / cutpoints for `FitResult` |
| `posterior_predictive_check` | `model` | one-shot PPC on agreement statistic (M0) |
| `FitResult` | `model` | dataclass for the M0 fit |
| `fit` | `model` | **DEPRECATED** alias for `fit_ordinal`; emits `DeprecationWarning` |
| `three_view_report` | `report` | direct + BTD + M0 side-by-side report |
| `print_three_view` | `report` | pretty-print the three-view report |

### Implementation files (one sentence each)

- `src/pairwise_rank/btd.py` (663 LOC) — the BTD model: fit,
  summarize, predict, direct_summary, the `pm.Potential`-based
  hand-rolled softmax that should become `pm.Categorical(logit_p=...)`.
- `src/pairwise_rank/model.py` (407 LOC) — the legacy M0 ordered
  logit: fit, summarize, posterior_predictive_check (the
  hand-rolled 5-category PPC is the largest custom-math block in
  the package).
- `src/pairwise_rank/protocol.py` (310 LOC) — verdict scales,
  Observation, schedule, run_tournament, JSONL I/O. No
  statistical code; small and in good shape.
- `src/pairwise_rank/report.py` (238 LOC) — three_view_report
  orchestrator and printer. No statistical code worth touching.
- `src/pairwise_rank/__init__.py` (75 LOC) — re-exports. Hosts
  `__all__` and the deprecation alias.
- `examples/synthetic.py` (120 LOC) — runnable 3-level pipeline
  with a deterministic ground-truth judge. Reproduces the
  ranking of a 4-candidate tournament.
- `examples/three_view.py` (81 LOC) — runs the three-view report
  on a 4-candidate synthetic 3-level tournament.

### Test files (one sentence each)

- `tests/test_btd.py` (12 tests, 194 LOC) — BTD invariants:
  strong-collapse, sign convention, zero-sum theta, P(best)
  sum, nu positivity, probability sum, position bias, direct
  counts.
- `tests/test_btd_predict.py` (11 tests, 193 LOC) —
  `predict_btd` per-cell behavior, position-neutral parity,
  sampler-diagnostics surfacing, backward-compat keys.
- `tests/test_model.py` (8 tests, 132 LOC) — M0 sign convention,
  cutpoint ordering / centering, posterior predictive check.
- `tests/test_protocol.py` (24 tests, 329 LOC) — verdict
  scale, observation schedule, JSONL round-trip, judge return
  shape.
- `tests/test_recovery.py` (1 test, 89 LOC) — end-to-end
  synthetic smoke test on the M0 / 5-level path.
- `tests/test_report.py` (9 tests, 121 LOC) — three_view_report
  structure, top-1 agreement, theta correlation between BTD and
  M0.
- `tests/test_v04.py` (16 tests, 441 LOC) — v0.4 default-protocol
  contract tests. **This file pins the [0, 2] `tournament_score`
  normalization — see §9 STOP condition.**

### Doc files

- `EXPERIMENT_DESIGN.md` (999 lines, committed) — methodology
  document. Already rewritten this session as a 19-section
  coherent document (commit `ba9bcea`). Section 1 covers
  estimands, §3 candidate-set validity, §5 the judge interface,
  §9 model choice, §10 diagnostics, §17 deployment validity /
  false evidence binding.
- `AGENTS.md` (250 lines, committed) — non-negotiable operating
  rules for any agent working in the repo. Read it before
  doing anything.
- `README.md` (334 lines, committed) — package overview.
- `METHODS_AUDIT.md` (720 lines, **untracked**) — this session's
  audit artifact. Detailed inventory, dependency decision
  table, parameterization crosswalk, deletion plan, projected
  LOC change.

## 5. Work completed in this session

### Completed (committed)

- `EXPERIMENT_DESIGN.md` was rewritten as a coherent 19-section
  methodology document. Six corrections applied per user review:
  three-layer estimand in §1, presentation-deployment rule in §4,
  full instrument surface in §5, direct-score formula in §8,
  Davidson / BTD model-selection language in §9, ESS corrected
  in §10, theta-not-comparable-across-fits in §18, two-trigger
  false-evidence-binding in §17, strategy/micro-type/artifact
  inference distinction in §14. **Committed: `ba9bcea`**.

### Completed (not committed; working tree only)

- `METHODS_AUDIT.md` (720 lines, untracked) — full read-only audit
  of every piece of custom statistical code in the package.
  Inventory, dependency decision table, projected deletion plan,
  STOP-condition flag for the `tournament_score` normalization
  bug, parameterization crosswalk, model correctness report
  (read-only), reproduction commands, open questions for the
  user. **Not committed. Not tested (does not affect runtime).**
  If the working tree is discarded before commit, this file is
  lost. Commit it before any tree reset.

### Attempted but abandoned

- PyMC execution (running the test suite to confirm a baseline
  pass/fail count). The sandbox repeatedly dropped pymc /
  pytest / numpyro / jax installs between commands, and the
  final test run before this handoff could not complete.
  Mitigation: the audit document is intentionally read-only,
  and the refactor is queued as a follow-up commit for a session
  with a stable env. **No code changes were made beyond
  `EXPERIMENT_DESIGN.md`.**

## 6. Statistical implementation audit so far

| component | current implementation | suspected issue | canonical replacement / reference | status |
|---|---|---|---|---|
| Davidson likelihood | `btd.py:161-192` builds a hand-rolled softmax (`pt.stack` / `pt.logsumexp` / subtraction) and injects it via `pm.Potential("y_obs_logp", pt.sum(log_probs[arange, ys]))`. Docstring claim "not exposed on every pytensor build" is inaccurate. | duplicate math, hand-rolled numerical stabilization, inaccurate docstring | `pm.Categorical("y", logit_p=logits, observed=ys)` (PyMC primitive; accepts `logit_p` since at least v5.6.0) | **HYPOTHESIS** — verified by reading the code; not yet refactored |
| BTD probability in `predict_btd` | `btd.py:549-556` repeats the same hand-rolled softmax with `np.maximum(np.maximum(a, b), c)` for numerical stability | third copy of the same model math | shared helper, used by fit / summarize / predict; or `scipy.special.softmax` | **HYPOTHESIS** |
| BTD probability in `summarize_btd` per-pair | `btd.py:339-353` same hand-rolled softmax | duplicate of predict_btd | shared helper | **HYPOTHESIS** |
| Categorical normalization | three copies of `softmax([a, b, c])` with hand-rolled stability | redundant, brittle to subtle changes | one shared helper | **HYPOTHESIS** |
| Zero-sum theta | `pm.ZeroSumNormal("theta", sigma=sigma_theta, shape=n)` (`btd.py:172`, `model.py:110`) | none — uses PyMC's maintained distribution | (already canonical) | **OK** |
| Ordered logistic | `pm.OrderedLogistic("y_obs", eta=eta, cutpoints=cutpoints, observed=ys)` (`model.py:122`) | none — uses PyMC's maintained distribution | (already canonical) | **OK** |
| Ordered cutpoints | `model.py:113-119` — `pm.Normal("cutpoint_gap_raw", 0, 0.7, shape=3)` then `pt.softplus` then `pt.cumsum` then zero-mean centering via `pm.Deterministic("cutpoints", k_uncentered - mean(k_uncentered))` | small, documented, identifiable transformation. Not a library primitive but correct and explicit. | (keep; document) | **OK** |
| Sampler invocation | `btd.py:235-241`, `model.py:163-169` both call `pm.sample(..., nuts_sampler="numpyro", ...)` explicitly. Current PyMC default (when `nutpie` is installed) is `nutpie`. | explicit non-default choice; bypasses PyMC's default selection | let PyMC pick; or use `pm.sample(..., nuts_sampler="nutpie", ...)` | **TODO** — benchmark during the refactor; do not silently change |
| Divergences | `btd.py:247-251` reads `idata.sample_stats["diverging"].sum().item()`; falls back to `0` if key missing. | silent fallback to 0 when key missing means "config problem" looks like "0 divergences". | read directly; fallback to `None` (signal "unknown") | **TODO** — fix the fallback during the refactor |
| R-hat / ESS / HDI | `btd.py:430-444` calls `az.summary(idata, var_names=[...], hdi_prob=hdi_prob)`. Per-parameter HDI via `az.hdi(...)`. | none — already delegated to ArviZ | (already canonical) | **OK** |
| Per-item rank / P(best) / P(top2) / expected rank | `btd.py:308-312, 314-324` and `model.py:267-283` — `np.argsort(-theta, axis=1)`, `np.where`, `mean`, in Python `for i in range(n)` loops. | O(n) Python loops with vectorizable numpy | vectorize using `np.argsort(np.argsort(-theta, axis=1), axis=1)` for ranks, `np.mean(np.argmax(theta, axis=1)[:, None] == np.arange(n), axis=0)` for P(best), broadcasting for pairwise P | **TODO** |
| Pairwise P(theta_i > theta_j) | `btd.py:327-331`, `model.py:235-243` — same O(n²) double loop with `(theta[:, i] > theta[:, j]).mean()`. | duplicate; loops; could be one shared vectorized helper | `np.mean(theta[:, :, None] > theta[:, None, :], axis=0)` | **TODO** |
| Direct-score normalization | `btd.py:641-648` — `(w + 0.5 * t) / (n - 1)` where `n` is the number of items. Produces a [0, 2] statistic for K=1, not [0, 1]. | **wrong denominator for a probability-like score**. Test `test_v04.py:388` pins `score["a"] = 1.5` for `W=2, T=2, N=3, K=1`; correct value with `2*K*(N-1) = 4` is `0.75`. | `(w + 0.5 * t) / (2 * K * (N - 1))` | **STOP CONDITION** — see §9 |
| Position-neutral prediction | `btd.py:271-275` (`_position_neutral_beta`) and `btd.py:525-528` (inline in `predict_btd`). | duplicated logic | one shared helper that returns the right beta array given `position_neutral` | **TODO** |
| P(best) | `btd.py:310`, `model.py:269` — `np.array([(np.argmax(theta, axis=1) == i).mean() for i in range(n)])`. | joint-event correct; just slow | vectorize | **OK semantically**; speed is minor |
| Hand-rolled 5-category probability in PPC | `model.py:381-386` — `p0 = expit(c0 - eta); p1 = expit(c1 - eta) - p0; ...` then `np.clip` and `rng.choice(5, p=probs)`. | the most custom-math-heavy block in the package | `pm.sample_posterior_predictive(idata, model=model, ...)`; aggregate the agreement statistic over the predictive samples | **TODO** |
| Strong-collapse count | `btd.py:111-118` (`_strong_count`), `btd.py:451-463` (`_btd_verdict_counts`), `btd.py:661-662` (inline in `direct_summary`). | three copies of the same count | one helper or `collections.Counter` | **TODO** |
| Deprecated alias | `model.py:189-220` `fit` emits `DeprecationWarning` and forwards to `fit_ordinal`. | redundant; only `test_model.py:11` calls it | keep through 0.5.0; mark for removal in 0.6.0 | **OK** for now; **TODO** to track deprecation timeline |

## 7. External library research already established

Conclusions already reached (verify against current official docs
before changing code):

- **`pm.Categorical(logit_p=...)` is the canonical PyMC primitive
  for a 3-outcome categorical with logits.** The current code's
  docstring claim "the latter is not exposed on every pytensor
  build" is inaccurate: `logit_p` is in the public API since at
  least PyMC 5.6.0. The next session should re-verify against the
  current docs.
- **`pm.OrderedLogistic` is the canonical ordered-logit.** Already
  used in `model.py`. Keep.
- **`pm.ZeroSumNormal` is the canonical sum-to-zero prior.** Already
  used. Keep.
- **`pm.sample_posterior_predictive` is the canonical PPC.** The
  current hand-rolled PPC in `model.py:posterior_predictive_check`
  should be replaced.
- **`pm.sample()` with the current PyMC default sampler** (nutpie
  when installed, otherwise PyMC NUTS) is the recommended path.
  The current code's explicit `nuts_sampler="numpyro"` may be
  unnecessarily opinionated; benchmark before changing.
- **ArviZ for diagnostics is correct** (`az.hdi`, `az.summary`).
  Keep.
- **bpcs (R + Stan) is reference-only**: not a runtime dep. Use
  as a one-time external validation if practical; not done in
  this session.
- **choix is not a substitute for Davidson ties.** The brief is
  explicit on this point; rejecting choix.
- **Bambi is not adopted** for M0. The M0 model is small enough
  that adding a wrapper layer is unjustified.

Runtime deps (current `pyproject.toml`): `pymc`, `pytensor`,
`arviz`, `numpy`, `scipy`. Dev dep: `pytest`. **No new
dependencies are recommended** by the audit.

Things considered and rejected: `choix`, `Bambi`, `pandas`,
`xarray` (transitive via ArviZ is fine; not a direct dep).

## 8. Exact Davidson parameterization

For an observation comparing left item `L` and right item `R`:

```
a_L = theta[L]
a_R = theta[R] + beta_right
nu  = exp(eta_tie)
```

Three unnormalized log weights:

```
log w_LEFT  = a_L
log w_TIE   = eta_tie + 0.5 * (a_L + a_R)
log w_RIGHT = a_R
```

Normalized:

```
P(LEFT)  = exp(a_L)                                / Z
P(TIE)   = exp(eta_tie + 0.5 * (a_L + a_R))        / Z
P(RIGHT) = exp(a_R)                                / Z
Z        = exp(a_L) + exp(eta_tie + 0.5*(a_L+a_R)) + exp(a_R)
```

with `theta ~ ZeroSumNormal(sigma=sigma_theta)`,
`sigma_theta ~ HalfNormal(1)`, `beta_right ~ Normal(0, 0.5)`,
`eta_tie ~ Normal(0, 1)`. `nu = exp(eta_tie) > 0` always; as
`eta_tie -> -inf`, the model reduces to Bradley-Terry. The
tie term is the Davidson **geometric-mean** form, not the
Rao-Kupper `(lambda_i + lambda_j)/2` form.

**Current code matches this parameterization exactly.** The
fit-side implementation is in `btd.py:170-191`; the predict-
side is in `btd.py:533-565`; the per-pair summary is in
`btd.py:339-353`. The three implementations are algebraically
identical but use different numerical implementations
(pytensor `pt.logsumexp` for fit, hand-rolled `np.maximum` for
NumPy sides). Do not change the math; consolidate into one
shared helper.

## 9. Known bugs / risks

| # | severity | issue | evidence | file / function | recommended next action |
|---|---|---|---|---|---|
| 1 | **high — STOP** | `direct_summary.tournament_score` denominator is `N - 1` instead of `2*K*(N-1)`. Produces a [0, 2] statistic, not a probability-like [0, 1] for K=1. | doc says [0,1]; code produces [0,2]; test `test_v04.py:388-390` asserts `score["a"] = 1.5` (the wrong value) and `test_v04.py:418` asserts `score["a"] = 2.0` (also wrong). | `btd.py:641-648`; tests `tests/test_v04.py:362-418`; doc `EXPERIMENT_DESIGN.md §8` | per task brief, STOP and resolve with the user before any model change. Recommended: option 1 — fix the code to match the doc (denominator `2*K*(N-1)`), update the two failing test values, add a [0, 1] range test. The `tournament_score` name implies [0, 1]; the field is used downstream as a probability. |
| 2 | medium | BTD likelihood hand-rolled in `pm.Potential` instead of `pm.Categorical(logit_p=...)`. | `btd.py:188-191` does `pt.stack` + `pt.logsumexp` + `pm.Potential("y_obs_logp", pt.sum(log_probs[arange, ys]))`. Docstring at `btd.py:43-46` says `pm.Categorical` is "not exposed on every pytensor build" — inaccurate. | `btd.py:161-192` | refactor: replace with `pm.Categorical("y", logit_p=logits, observed=ys)`. Update docstring. |
| 3 | medium | BTD probability computed in three places (fit, summarize per-pair, predict). | identical formulas at `btd.py:184-186`, `btd.py:341-353`, `btd.py:549-556`. | `btd.py` (3 sites) | refactor: one shared `_davidson_log_probs` helper used by all three. |
| 4 | medium | PPC hand-rolled instead of `pm.sample_posterior_predictive`. | `model.py:376-391` builds 5-category probability from `expit(c_k - eta) - expit(c_{k-1} - eta)`, clips, and `rng.choice(5, p=probs)`. | `model.py:334-407` `posterior_predictive_check` | refactor: use `pm.sample_posterior_predictive`; aggregate the agreement statistic over predictive samples. |
| 5 | low | `pm.Potential` docstring is inaccurate. | `btd.py:43-46` | `btd.py:43-46` | fix the docstring as part of refactor 2. |
| 6 | low | O(n) and O(n²) Python loops over items in summary code. | `btd.py:308-312, 314-324, 327-331`; `model.py:235-243, 267-283` | `btd.py`, `model.py` | vectorize; perf is not the issue, clarity is. |
| 7 | low | Three copies of strong-collapse count. | `btd.py:111-118, 451-463, 661-662` | `btd.py` | dedupe with `collections.Counter` or one helper. |
| 8 | low | Divergences fallback to 0 when key missing. | `btd.py:250-251` — `except (KeyError, AttributeError): n_divergences = 0` | `btd.py:247-251` | change to `None` to signal "unknown" rather than "0". |
| 9 | low | Sampler backend explicit (`nuts_sampler="numpyro"`) when PyMC default (nutpie) may be preferred. | `btd.py:237`, `model.py:165` | `btd.py`, `model.py` | benchmark during refactor; prefer the simpler default. |
| 10 | low | `fit` deprecation alias. | `model.py:189-220` emits `DeprecationWarning`; only `test_model.py:11` calls it. | `model.py` | keep through 0.5.0; mark for removal in 0.6.0. |
| 11 | low | Cross-fit theta comparisons. | not currently done in code, but the doc warns against it. The audit only inspects the doc; no test failure. | `EXPERIMENT_DESIGN.md §18` | already documented; no code change needed. |
| 12 | low | No external reference validation against bpcs / Stan. | not run in this session. | n/a | one-time dev script; not a runtime dep; deferred. |
| 13 | low | `pm.Potential` is the only hand-rolled `pm.Potential` in the package — no other custom PyMC primitives. | grep `pm.Potential` in `src/pairwise_rank/*.py` returns 2 hits, both in `btd.py` (one in the model, one in the docstring). | `btd.py` | covered by risk 2. |

## 10. Tests and validation state

- **Test command**:
  ```
  PYTHONPATH=src python3 -m pytest tests/ -q --tb=line
  ```
- **Test count**: 81 (verified by `grep -c "^def test_"`
  per file: 12 + 11 + 8 + 24 + 1 + 9 + 16 = 81).
- **Current pass/fail count**: **UNKNOWN**. The sandbox kept
  dropping `pymc` / `pytest` / `numpyro` between commands; the
  test suite could not be completed reliably in this session.
  The last successful run before the env instability showed
  `43 failed, 32 passed, 9 warnings, 6 errors` — but those
  failures were largely environment (missing `numpyro`,
  missing `jax`, missing `pytest`), not code. After installing
  the dependencies, the test suite was not run successfully
  again before the session was halted.
- **The next session's first action should be to re-run the
  test suite** to obtain a clean baseline pass/fail count.
- **Synthetic recovery test**: `tests/test_recovery.py::test_synthetic_end_to_end_recovers_ordering`
  exercises the M0 5-level path against a synthetic 4-candidate
  tournament with deterministic ground truth. Asserts the
  recovered ranking matches and P(best) > 0.9 for the strongest
  item. No equivalent end-to-end recovery test for BTD /
  3-level exists yet.
- **Invariant tests present** (per `tests/test_btd.py` and
  `tests/test_v04.py`): strong-collapse, sign convention,
  zero-sum theta at every draw, P(best) sum-to-1, nu positivity,
  probability sum, position-bias sign, position-neutral parity.
  Per-task-brief items 21-24 (legacy 5-level collapse, malformed
  verdicts, per-cell probability sum) are covered.
- **Invariant tests missing** (per task brief items 17-19, 25-28):
  P(best) is joint draw-by-draw; expected-rank bounds; top-k
  monotonicity; pairwise probability matrix neutral symmetry;
  synthetic recovery for known theta ordering, beta_right sign,
  and tie tendency; no accidental dependence on candidate label
  order. The next session should add these.
- **External reference comparison**: not done. Deferred.
- **Benchmark numbers**: none collected.

## 11. Documentation state

`EXPERIMENT_DESIGN.md` is the methodology document. It was
rewritten this session as a coherent 19-section document
covering: estimand (§1), construct (§2), candidate-set
validity (§3), presentation fidelity (§4), judge interface as
instrument (§5), prompt design (§6), balanced protocol (§7),
direct + global inference (§8), model choice (§9),
diagnostics (§10), reasoning audit (§11), matched ablation
(§12), discovery-confirmation-stop (§13), coarse-to-fine
search (§14), selection leakage (§15), artifact-vs-strategy
inference (§16), deployment validity / false evidence
binding (§17), reporting / claim hygiene (§18), experimental
checklist (§19). **Committed at `ba9bcea`.**

Important current conceptual direction in the doc:

- three-level / direct / BTD is the current normal path
- matched ablation is a core method, not an appendix
- coarse-to-fine search must retain a sibling incumbent to
  detect abstraction collapse
- artifact-level winners do not imply category-level causal
  claims
- deployment validity / false evidence binding is a separate
  gate from ranking validity
- tool metadata and the complete judge interface are part of
  the experimental instrument
- selection leakage / proxy inversion / anti-signaling need
  explicit treatment
- no personal profile experiments or strings belong in the
  public repo

The rewrite is committed. There is no uncommitted diff to
`EXPERIMENT_DESIGN.md`. The one uncommitted file in the working
tree is `METHODS_AUDIT.md` (see §5).

## 12. What went wrong in this session

The next agent needs to know what NOT to repeat.

- **The audit was correctly bounded to read-only analysis.** That
  worked. Do not abandon that discipline.
- **The refactor was correctly deferred** to a follow-up commit
  pending a stable env. That was right.
- **The blocker** was the sandbox repeatedly losing installed
  packages between commands (`pymc`, `pytest`, `numpyro`, `jax`,
  `jaxlib`). After each successful install, the next command
  sometimes could not import them. This caused thrashing between
  `pip install` and `pytest`, neither of which produced stable
  results.
- **A clean refactor was attempted once** ("Install the missing
  libs. Start over. The sandbox restarts often, I have no control
  over it.") and one `pytest tests/` call appeared to succeed
  in timing but was reported as "caller aborted" on
  transmission back. The user then said "stop" before the
  refactor began.
- **What the next session should do differently**:
  1. Install the dependencies once at the top of the session
     (not reactively after each drop).
  2. Run the test suite as the second action (the first is
     `git status`).
  3. If the test suite fails for environment reasons, fix the
     environment, not the code.
  4. If the test suite passes, do not begin the refactor on the
     same turn as running the test suite. The test run is
     observation; the refactor is action.
  5. Resolve the §9 STOP condition (tournament_score [0, 2]
     normalization) with the user *before* any model change.
  6. Treat `METHODS_AUDIT.md` as the existing plan. Do not
     re-audit from scratch. Read it, verify the findings still
     hold against the current code, and execute.
  7. Do not retry the `pm.Categorical(logit_p=...)` refactor
     more than once if it fails. The API has been stable since
     v5.6.0; if it fails, the issue is the local env, not the
     API.

## 13. Recommended next-session plan

1. **Read this handoff and inspect the working tree directly.**
   Evidence required: `git status` is clean except for the
   untracked `METHODS_AUDIT.md`.
2. **Install dependencies once.**
   ```
   pip install --break-system-packages --index-url https://pypi.org/simple/ \
     pymc pytensor arviz numpy scipy nutpie numpyro jax jaxlib pytest
   ```
3. **Run the test suite as observation, not as a gate.**
   ```
   PYTHONPATH=src python3 -m pytest tests/ -q --tb=line
   ```
   Record pass/fail. Do not change any code in this step.
4. **Re-verify the audit findings against the current source.**
   Open `btd.py`, `model.py`, `report.py`, `protocol.py` and
   confirm that the inventory in `METHODS_AUDIT.md` §A is
   accurate. Evidence required: line numbers and
   `pm.Potential` / `pt.stack` / `np.exp` / `np.maximum`
   counts match.
5. **Re-verify the `pm.Categorical(logit_p=...)` API** from
   current PyMC official docs. The audit relied on the
   web-fetched docs from this session; re-verify before
   refactoring.
6. **Resolve the §9 STOP condition with the user.** Fix the
   denominator, update the two failing tests, add a [0, 1]
   range test. This is a prerequisite for any model change.
7. **Run the synthetic example to establish a behavior
   baseline.**
   ```
   PYTHONPATH=src python3 examples/synthetic.py
   PYTHONPATH=src python3 examples/three_view.py
   ```
   Record the recovered ranking and the position-effect sign.
   Evidence required: reproducible output across runs.
8. **Replace the BTD hand-rolled softmax in `_build_btd_model`
   with `pm.Categorical(logit_p=logits, observed=ys)`.** Re-run
   the test suite. Re-run the synthetic example. Confirm the
   recovered ranking and position effect sign are unchanged
   (within MC noise).
9. **Add one shared `_davidson_log_probs` helper** and use it
   in `summarize_btd` and `predict_btd`. Replace the
   `np.maximum(np.maximum(a, b), c)` pattern with
   `scipy.special.softmax` (or the helper's own stable
   implementation). Re-run tests.
10. **Replace the hand-rolled PPC in
    `posterior_predictive_check` with `pm.sample_posterior_predictive`.**
    Re-run tests.
11. **Vectorize** the per-item / per-pair summary loops in
    `summarize_btd` and `summarize`. Re-run tests.
12. **Add the missing invariant tests** (Phase 10 of the audit
    brief items 17-19, 25-28). These pin the model correctness
    that the refactor must preserve.
13. **Compare against an independent Davidson reference if
    practical** (bpcs in R + Stan, in a dev environment). If
    not practical, document that the comparison was not
    performed and the canonical PyMC API is the substitute.
14. **Delete redundant code** (Phase 12 of the audit brief).
    Projected -273 LOC (-16%) across the package.
15. **Update docs** to match the final implementation. The
    doc already describes the desired model. Confirm the
    parameterization crosswalk in `METHODS_AUDIT.md` §G still
    holds.
16. **Commit only after all tests pass and the synthetic
    example reproduces.** One commit, message: "Refactor BTD
    model to use maintained PyMC primitives; fix direct_score
    normalization; consolidate model math".

Each step has explicit "evidence required before moving on" so
the next session does not drift.

## 14. Do-not-do list

The new agent must NOT:

- add features
- build a framework / plugin / registry / factory / provider
  abstraction / new base class
- write a custom HMC / NUTS / leapfrog / mass-matrix / step-size
  / proposal / acceptance / chain-management code
- replace the Davidson model with plain Bradley-Terry
- add dependencies just because they exist
- compare theta magnitudes across unrelated fits
- treat reasoning traces as causal ground truth
- bless divergences using an arbitrary percentage threshold
- silently change priors
- silently break the public API
- rewrite working code before understanding its current
  semantics
- preserve custom math merely because it already exists
- perform broad refactors before establishing reference tests
- begin the refactor on the same turn as the test run
- retry `pm.Categorical(logit_p=...)` integration more than
  once if it fails; the API has been stable since v5.6.0
- touch the deprecated `fit` alias without a deprecation
  timeline
- introduce a new sampler backend without benchmarking

## 15. Commands for the next agent

```
# Enter repo
cd /run/csi/mount-root/nas/eab0d61a99b6696edb3d2aff87b585e8/pairwise-rank

# Inspect status / diff
git status --short
git diff --stat
git log --oneline -8

# Install deps (one shot, at the top of the session)
pip install --break-system-packages --index-url https://pypi.org/simple/ \
  pymc pytensor arviz numpy scipy nutpie numpyro jax jaxlib pytest

# Run tests (do not change code in this step)
PYTHONPATH=src python3 -m pytest tests/ -q --tb=line

# Inspect relevant files
ls -la
cat pyproject.toml
cat src/pairwise_rank/__init__.py
head -80 src/pairwise_rank/btd.py
head -50 src/pairwise_rank/model.py

# Read the audit document
head -100 METHODS_AUDIT.md
grep -n "pm.Potential\|pt.stack\|pt.logsumexp\|np.maximum" src/pairwise_rank/btd.py

# Synthetic baselines
PYTHONPATH=src python3 examples/synthetic.py
PYTHONPATH=src python3 examples/three_view.py

# After refactor: re-run the synthetic examples and confirm
# the recovered ranking and position effect sign are unchanged.

# Commit (one commit, only after all tests pass)
git add src/pairwise_rank/btd.py src/pairwise_rank/model.py \
        tests/test_v04.py METHODS_AUDIT.md 2>/dev/null
git status
git -c http.sslVerify=false commit -m "..."
```

## 16. Minimal context prompt for the fresh session

```
Read CONTEXT_HANDOFF.md completely before making changes.

You are continuing an audit-and-simplification pass on a small
Python library called `pairwise-rank` (a Bayesian ordinal
paired-comparison model using PyMC + ArviZ). The previous session
delivered a read-only audit (METHODS_AUDIT.md) and committed a
rewrite of EXPERIMENT_DESIGN.md. The refactor was deferred because
the sandbox repeatedly dropped installed dependencies mid-session.

Working tree state: one untracked file (METHODS_AUDIT.md) on top
of commit `ba9bcea`. The package and tests are otherwise clean.

Your job, in this order:

1. Inspect the repo. Confirm `git status` is clean except for
   METHODS_AUDIT.md. Confirm the audit document is what you want
   to execute.
2. Install dependencies once at the top of the session
   (pymc, pytensor, arviz, numpy, scipy, nutpie, numpyro, jax,
   jaxlib, pytest).
3. Run the test suite to obtain a clean baseline pass/fail
   count. Record it. Do not change code in this step.
4. Resolve the STOP condition in CONTEXT_HANDOFF.md §9 with the
   user before any model change. The direct_summary tournament
   score denominator is wrong; the test pins the wrong value.
5. Re-verify the PyMC API claims in METHODS_AUDIT.md §B against
   current official docs before refactoring.
6. Execute the refactor in the order given in
   CONTEXT_HANDOFF.md §13. Do not reorder. Do not batch.
7. After each refactor step, re-run the test suite and the
   synthetic example (examples/synthetic.py and
   examples/three_view.py). The recovered ranking and
   position-effect sign must be unchanged (within MC noise).
8. Use current official documentation to verify any API you
   are not certain about. Do not rely on memory or on this
   session's web fetches alone.
9. Preserve the exact statistical semantics enumerated in
   CONTEXT_HANDOFF.md §3. Especially: do not replace Davidson
   with plain Bradley-Terry; do not compare theta magnitudes
   across fits; do not bless divergences; do not treat
   reasoning traces as causal.
10. Prefer deletion and canonical library calls over new
    abstractions. The target is a smaller, more boring codebase
    that someone familiar with PyMC can read in a few minutes.
11. Do not add features, frameworks, plugins, registries, or
    new base classes. This project is a small tool, not a
    platform.
12. Show evidence before changing model mathematics. The
    exact Davidson parameterization is in CONTEXT_HANDOFF.md §8.
    The current code matches it. Any change must preserve it.
13. Commit only when all tests pass and the synthetic example
    reproduces. One commit, descriptive message.

Stop conditions (report to user before proceeding):

- Existing tests encode contradictory semantics (already known
  for the tournament_score normalization; resolve first).
- A proposed change would alter priors or model semantics.
- The PyMC API you intend to use has changed since the docs
  read by the previous session.
- A reference validation reveals a real statistical
  discrepancy.
- Public API would need to break.
```

---

## Quality bar

The handoff distinguishes:

- **FACT**: verified from repository state, source code, pyproject,
  and `git` output in this session. Includes: HEAD commit, file
  LOC, public API list, test count, git state.
- **DECISION**: methodological choices already made and committed.
  Includes: 3-level default, Davidson geometric-mean tie term,
  `pm.ZeroSumNormal`, `pm.OrderedLogistic`, position-neutral
  with `beta_right = 0`, etc. These live in `EXPERIMENT_DESIGN.md`
  and `btd.py` / `model.py`.
- **HYPOTHESIS**: suspected problems not yet verified in this
  session. The audit's findings about duplicated BTD math,
  hand-rolled softmax, hand-rolled PPC, divergence fallback to
  0, etc. are all HYPOTHESIS until the refactor is run and the
  synthetic example reproduces.
- **TODO**: work the next session must perform. This is the
  Phase 0 → Phase 15 list in `METHODS_AUDIT.md` and the §13
  recommended next-session plan above.

The `tournament_score` [0, 2] bug is the only known case where
the repo (code and test) disagrees with the methodology (doc).
Resolve that first.
