# Context handoff

## 1. Mission

Aggressive audit and simplification of `pairwise-rank` (a small
Python library for Bayesian ordinal paired-comparison ranking).
Remove bespoke statistical machinery wherever PyMC, ArviZ, scipy,
or other maintained libraries already supply a canonical
implementation. Davidson / Bradley-Terry-Davidson semantics must
remain exact; the 3-level LEFT/TIE/RIGHT protocol must remain the
default; direct counterbalanced evidence must remain primary for
small matched head-to-heads. Not a feature expansion.

## 2. Repository state

- **Path**: `/run/csi/mount-root/nas/eab0d61a99b6696edb3d2aff87b585e8/pairwise-rank`
- **Branch**: `main` @ `ba9bcea` (clean)
- **Worktree for this work**: `.worktrees/audit-handoff/` on `feature/audit-and-handoff` @ `befa8d3`
- **Working tree**:
  - `main` clean except `.worktrees/` (gitignored? no — gitignored: `__pycache__/`, `build/`, `*.egg-info/`, `.pytest_cache/`; `.worktrees/` is NOT in `.gitignore`)
  - `feature/audit-and-handoff` has `METHODS_AUDIT.md` and `CONTEXT_HANDOFF.md` committed
- **Recent commits** (most recent first):
  ```
  befa8d3 Add methodology audit and context handoff for next session   (feature/audit-and-handoff)
  ba9bcea Rewrite experimental design methodology
  64b341e EXPERIMENT_DESIGN.md §18: coarse-to-fine with sibling-incumbent retention
  4f61102 Document matched-ablation methodology
  35d9504 v0.4.2: Add AGENTS.md and tool-metadata-as-rubric design lesson
  bdb3ec6 Add per-cell BTD predictions and sampler diagnostics
  bf86796 Demote 5-level ordinal model; BTD + 3-level protocol is the new default
  ```
- **Do not lose**:
  - `METHODS_AUDIT.md` (720 lines, audit plan with inventory + dependency table)
  - `CONTEXT_HANDOFF.md` (this file)
  - `EXPERIMENT_DESIGN.md` (999 lines, committed at `ba9bcea`)
  - `tests/test_v04.py:362-418` (the two tests that pin the [0, 2] normalization — must be updated as part of the same commit that fixes the code)

## 3. Methodology contract — semantics that must not silently change

- 3-level default: `LEFT / TIE / RIGHT`. Legacy 5-level `LEFT_STRONG` / `RIGHT_STRONG` collapse to ordinary wins/losses on ingest.
- Direct counterbalanced tallies are primary for small matched head-to-heads; BTD is the global cross-check.
- Davidson / Bradley-Terry-Davidson is the normal global model for multi-candidate LEFT/TIE/RIGHT tournaments.
- Tie term is the **Davidson geometric-mean** form `nu * sqrt(lambda_i * lambda_j)`, not Rao-Kupper.
- `beta_right` models right-slot / order effect. `beta_right > 0` means right-slot advantage.
- Position-neutral prediction uses `beta_right = 0` (predictions only; posterior unchanged).
- `theta ~ pm.ZeroSumNormal(sigma=sigma_theta)`. `theta` is field-relative. **θ magnitudes from different fits are not comparable.**
- `P(best)` is the joint posterior event: fraction of draws in which the candidate is the argmax-θ item. Not a softmax of θ means.
- M0 ordered logistic is legacy / special-purpose for genuinely ordinal intensity (5-level data). New code uses BTD.
- Divergences are geometry failures. No acceptable fraction. R-hat ≈ 1 is necessary, not sufficient. ESS is a Monte Carlo efficiency diagnostic, not a coverage measure.
- Reasoning traces are post-hoc audit metadata, not model inputs and not causal explanations.
- Direct-vs-global strain is not automatically a cycle. **A cycle requires ≥ 3 candidates.**
- HDI including zero is "not enough data", not "no bias". HDI excluding zero does not imply the ranking is trustworthy if the fit is broken.
- Tool metadata (function name, description, parameter names/descriptions/enum, `tool_choice`, message structure, reasoning setting, model/provider config, rendering config) is part of the instrument. Changing any of these is a prompt change.

**Where repo and methodology disagree**: `direct_summary.tournament_score` is [0, 2] in the code, [0, 1] in the doc. The test pins the [0, 2] value. STOP condition. See §9.

## 4. Public API

From `src/pairwise_rank/__init__.py`:

| symbol | role |
|---|---|
| `fit_btd` | **default** probabilistic model (3-level BTD). |
| `summarize_btd` | per-item / pairwise / diagnostics; supports `position_neutral`. |
| `predict_btd` | per-cell, orientation-aware BTD likelihood. |
| `direct_summary` | W/L/T counts + tournament score (no model). |
| `BTDFitResult` | dataclass: `theta_draws`, `beta_right_draws`, `sigma_theta_draws`, `eta_tie_draws`, `nu_draws`, `divergences`. |
| `fit_ordinal` | **legacy** 5-level ordered logit. |
| `summarize` | M0 summary (signature-compatible with `summarize_btd`). |
| `posterior_predictive_check` | one-shot PPC on agreement statistic. **Hand-rolled 5-category probability; replace with `pm.sample_posterior_predictive`.** |
| `FitResult` | M0 dataclass. |
| `fit` | **DEPRECATED** alias for `fit_ordinal`; emits `DeprecationWarning`. Only `tests/test_model.py:11` calls it. |
| `three_view_report`, `print_three_view` | direct + BTD + M0 report. |
| `Observation`, `observation_key`, `make_schedule`, `run_tournament` | protocol + scheduling + dedup. |
| `VERDICT_LEVELS`, `VERDICT_LEVELS_5`, `DEFAULT_VERDICT_LEVELS` | 3-level / 5-level label tuples. |
| `verdict_to_code`, `code_to_verdict`, `collapse_to_3_level` | verdict mapping; `collapse_to_3_level` is the single STRONG-collapse helper. |
| `save_observations_jsonl`, `load_observations_jsonl` | JSONL I/O. |

## 5. File layout

```
src/pairwise_rank/
  __init__.py    75 LOC   re-exports + __all__
  btd.py        663 LOC   BTD model + direct_summary — primary refactor target
  model.py      407 LOC   M0 ordered logit + hand-rolled PPC — secondary
  protocol.py   310 LOC   verdict scale, Observation, schedule, JSONL — in good shape
  report.py     238 LOC   three_view_report, print_three_view — in good shape

tests/
  test_btd.py            12 tests   BTD invariants
  test_btd_predict.py    11 tests   predict_btd, position-neutral parity
  test_model.py           8 tests   M0 invariants
  test_protocol.py       24 tests   verdict scale, schedule, JSONL
  test_recovery.py        1 test    M0 5-level end-to-end recovery smoke
  test_report.py          9 tests   three_view_report
  test_v04.py            16 tests   v0.4 contract; PIN THE [0,2] BUG

examples/
  synthetic.py    120 LOC   runnable 3-level pipeline
  three_view.py    81 LOC   runnable three-view report
```

Total: 81 tests collected. Pre-audit baseline: 32 passing, 43 failing,
6 errors — those failures were largely env (missing numpyro / jax /
pytest), not code. **Current pass/fail count UNKNOWN in this session**
because the sandbox repeatedly dropped installed packages. The next
session's first action should be `pip install` (one shot) then
`PYTHONPATH=src python3 -m pytest tests/ -q --tb=line` to establish
a clean baseline.

## 6. Statistical implementation audit

(FACT: verified by reading source. HYPOTHESIS: not yet exercised end-to-end. TODO: action the next session must take.)

| component | current | issue | canonical | status |
|---|---|---|---|---|
| Davidson likelihood | `btd.py:161-192` hand-rolled `pt.stack` + `pt.logsumexp` + `pm.Potential("y_obs_logp", pt.sum(log_probs[arange, ys]))`. Docstring at `btd.py:43-46` says `pm.Categorical` is "not exposed on every pytensor build" — inaccurate. | hand-rolled softmax; inaccurate docstring | `pm.Categorical("y", logit_p=logits, observed=ys)` (PyMC primitive, stable since v5.6.0) | TODO |
| BTD probability in `predict_btd` | `btd.py:549-556` repeats the same softmax with `np.maximum(np.maximum(a, b), c)` for stability | third copy of model math | one shared helper; or `scipy.special.softmax` | TODO |
| BTD probability in `summarize_btd` per-pair | `btd.py:339-353` | duplicate of predict_btd | shared helper | TODO |
| Categorical normalization | three hand-rolled softmax sites | redundant, brittle | one shared helper | TODO |
| Zero-sum θ | `pm.ZeroSumNormal("theta", sigma=sigma_theta, shape=n)` at `btd.py:172`, `model.py:110` | none | (already canonical) | FACT (OK) |
| Ordered logistic | `pm.OrderedLogistic("y_obs", eta=eta, cutpoints=cutpoints, observed=ys)` at `model.py:122` | none | (already canonical) | FACT (OK) |
| Ordered cutpoints | `model.py:113-119` — `pm.Normal("cutpoint_gap_raw", 0, 0.7, shape=3)` then `pt.softplus` then `pt.cumsum` then zero-mean centering | small, explicit, documented identification. Not a library primitive but correct. | (keep; document) | FACT (OK) |
| Sampler invocation | `btd.py:235-241`, `model.py:163-169` — both call `pm.sample(..., nuts_sampler="numpyro", ...)` explicitly. Current PyMC default (when `nutpie` is installed) is `nutpie`. | explicit non-default; bypasses PyMC's default selection | let PyMC pick; or `nuts_sampler="nutpie"` | TODO — benchmark during refactor |
| Divergences | `btd.py:247-251` reads `idata.sample_stats["diverging"].sum().item()`; falls back to `0` if key missing | silent fallback to 0 means "config problem" looks like "0 divergences" | fallback to `None` ("unknown") | TODO |
| R-hat / ESS / HDI | `btd.py:430-444` calls `az.summary(idata, var_names=[...], hdi_prob=hdi_prob)`. Per-parameter HDI via `az.hdi(...)`. | none | (already canonical) | FACT (OK) |
| Per-item rank / P(best) / P(top2) / expected rank | `btd.py:308-312, 314-324`; `model.py:267-283` — `np.argsort(-theta, axis=1)`, `np.where`, `mean`, in `for i in range(n)` loops | O(n) Python loops; vectorizable | one-liners with numpy broadcasting | TODO |
| Pairwise P(θ_i > θ_j) | `btd.py:327-331`, `model.py:235-243` — same O(n²) double loop, same code in two files | duplicate; loops | one shared vectorized helper | TODO |
| Direct-score normalization | `btd.py:641-648` — `(w + 0.5 * t) / (n - 1)`. Produces [0, 2] for K=1. | **wrong denominator for a probability-like score** | `(w + 0.5 * t) / (2 * K * (N - 1))` | **STOP** — see §9 |
| Position-neutral prediction | `btd.py:271-275` (`_position_neutral_beta`) and `btd.py:525-528` (inline) | duplicated logic | one shared helper | TODO |
| P(best) | `btd.py:310`, `model.py:269` — `np.array([(np.argmax(theta, axis=1) == i).mean() for i in range(n)])` | joint-event correct; slow | vectorize | FACT (semantics OK); speed minor |
| 5-category probability in PPC | `model.py:381-386` — `p0 = expit(c0 - eta); p1 = expit(c1 - eta) - p0; ...` then `np.clip` and `rng.choice(5, p=probs)` | the most custom-math-heavy block in the package | `pm.sample_posterior_predictive`; aggregate agreement over predictive samples | TODO |
| Strong-collapse count | `btd.py:111-118`, `btd.py:451-463`, `btd.py:661-662` — three copies | duplicate | one helper or `collections.Counter` | TODO |
| `fit` deprecation alias | `model.py:189-220` emits `DeprecationWarning`; only `test_model.py:11` calls it | redundant | keep through 0.5.0; mark for removal in 0.6.0 | TODO (deprecation timeline) |

## 7. External library research

Conclusions already reached (verify against current official docs
before refactoring):

- **`pm.Categorical(logit_p=...)` is canonical.** Stable since PyMC 5.6.0.
- **`pm.OrderedLogistic` is canonical.** Already used in `model.py`.
- **`pm.ZeroSumNormal` is canonical.** Already used.
- **`pm.sample_posterior_predictive` is canonical.** Replace the hand-rolled M0 PPC.
- **`pm.sample()` default (nutpie when installed, else PyMC NUTS).** The explicit `nuts_sampler="numpyro"` is unnecessarily opinionated; benchmark.
- **ArviZ for diagnostics** (`az.hdi`, `az.summary`) — correct; keep.
- **bpcs (R + Stan)** — reference-only, not runtime. One-time external validation if practical; not done in this session.
- **choix** — not a substitute for Davidson ties. Rejected.
- **Bambi** — wrapper layer over a small model. Rejected.

Runtime deps (current `pyproject.toml`): `pymc`, `pytensor`,
`arviz`, `numpy`, `scipy`. Dev: `pytest`. **No new dependencies
recommended** by the audit.

## 8. Exact Davidson parameterization

For left item `L`, right item `R`:

```
a_L = theta[L]
a_R = theta[R] + beta_right
nu  = exp(eta_tie)
log w_LEFT  = a_L
log w_TIE   = eta_tie + 0.5 * (a_L + a_R)
log w_RIGHT = a_R
P(outcome) = exp(log w) / Z,  Z = sum of exp(log w).
```

Priors: `theta ~ ZeroSumNormal(sigma=sigma_theta)`,
`sigma_theta ~ HalfNormal(1)`, `beta_right ~ Normal(0, 0.5)`,
`eta_tie ~ Normal(0, 1)`. `nu = exp(eta_tie) > 0` always; as
`eta_tie -> -inf`, the model reduces to Bradley-Terry. The tie
term is the **Davidson geometric-mean** form, not Rao-Kupper.

**Current code matches exactly.** Fit: `btd.py:170-191`. Predict:
`btd.py:533-565`. Per-pair summary: `btd.py:339-353`. The three
implementations are algebraically identical but use different
numerical implementations (pytensor `pt.logsumexp` for fit;
hand-rolled `np.maximum` for NumPy sides). Do not change the math;
consolidate into one shared helper.

## 9. Known bugs / risks

| # | sev | issue | evidence | location | action |
|---|---|---|---|---|---|
| 1 | **STOP** | `direct_summary.tournament_score` denominator is `N - 1`, producing [0, 2] not [0, 1]. Doc says [0, 1]; test pins [0, 2]. | `btd.py:641-648`; `tests/test_v04.py:388-390, 418` (asserts `score["a"] = 1.5` and `2.0` — wrong); `EXPERIMENT_DESIGN.md §8` | `btd.py`, `tests/test_v04.py` | resolve with the user before any model change. **Recommended**: fix code to match doc (denominator `2*K*(N-1)`), update the two failing test values, add a [0, 1] range test. The field name implies [0, 1] and is used downstream as a probability. |
| 2 | med | BTD likelihood hand-rolled instead of `pm.Categorical(logit_p=...)` | `btd.py:188-191`; docstring at `btd.py:43-46` is inaccurate | `btd.py:161-192` | replace with `pm.Categorical("y", logit_p=logits, observed=ys)`; fix docstring |
| 3 | med | BTD probability computed in three places | identical formulas at `btd.py:184-186`, `btd.py:341-353`, `btd.py:549-556` | `btd.py` (3 sites) | one shared `_davidson_log_probs` helper |
| 4 | med | PPC hand-rolled instead of `pm.sample_posterior_predictive` | `model.py:376-391` | `model.py:334-407` | replace |
| 5 | low | `pm.Potential` docstring inaccurate | `btd.py:43-46` | `btd.py:43-46` | fix during refactor 2 |
| 6 | low | O(n) and O(n²) Python loops in summary code | `btd.py:308-312, 314-324, 327-331`; `model.py:235-243, 267-283` | both | vectorize; perf minor, clarity matters |
| 7 | low | Three copies of strong-collapse count | `btd.py:111-118, 451-463, 661-662` | `btd.py` | dedupe |
| 8 | low | Divergences fallback to 0 on missing key | `btd.py:250-251` | `btd.py:247-251` | fallback to `None` ("unknown") |
| 9 | low | Explicit `nuts_sampler="numpyro"` when PyMC default may be preferred | `btd.py:237`, `model.py:165` | both | benchmark during refactor |
| 10 | low | `fit` deprecation alias still present | `model.py:189-220`; only `test_model.py:11` calls it | `model.py` | keep through 0.5.0; mark for 0.6.0 |
| 11 | low | No external reference validation against bpcs/Stan | not run this session | n/a | one-time dev script; deferred |

## 10. Tests and validation state

- **Test command**: `PYTHONPATH=src python3 -m pytest tests/ -q --tb=line`
- **Test count**: 81 (12+11+8+24+1+9+16, verified by `grep -c "^def test_"`).
- **Pass/fail count**: UNKNOWN. Sandbox dropped pymc/pytest/numpyro repeatedly. Last successful run: `43 failed, 32 passed, 9 warnings, 6 errors` — failures were env (missing numpyro / jax / pytest), not code. The next session must establish a clean baseline before refactoring.
- **Invariant tests present**: strong-collapse, sign convention, zero-sum θ, P(best) sum, nu positivity, probability sum, position-bias sign, position-neutral parity, M0 cutpoint ordering/centering.
- **Invariant tests missing** (per audit brief): P(best) is joint draw-by-draw; expected-rank bounds; top-k monotonicity; pairwise probability matrix neutral symmetry; synthetic recovery for known θ ordering, beta_right sign, and tie tendency; no accidental dependence on candidate label order.
- **External reference comparison**: not run. Deferred.
- **Benchmark numbers**: none collected.

## 11. Documentation state

- `EXPERIMENT_DESIGN.md` (999 lines) — committed at `ba9bcea`. 19-section coherent methodology rewrite covering estimand, construct, candidate validity, presentation fidelity, judge interface, prompt design, balanced protocol, direct + global inference, model choice, diagnostics, reasoning audit, matched ablation, discovery-confirmation-stop, coarse-to-fine, selection leakage, artifact-vs-strategy inference, deployment validity, claim hygiene, experimental checklist. The rewrite is committed; no uncommitted diff. The methodology direction is the canonical reference for the refactor.
- `AGENTS.md` (250 lines) — non-negotiable operating rules. Read before any work.
- `METHODS_AUDIT.md` (720 lines) — this session's audit artifact, in the worktree on `feature/audit-and-handoff` @ `befa8d3`. Inventory, dependency table, deletion plan, parameterization crosswalk, model correctness report (read-only), reproduction commands, open questions.
- `CONTEXT_HANDOFF.md` (this file) — also in the worktree.

Important current conceptual direction in `EXPERIMENT_DESIGN.md`:
3-level/direct/BTD is the current normal path; matched ablation is
core; coarse-to-fine search must retain a sibling incumbent;
artifact-level winners do not imply category-level causal claims;
deployment validity / false evidence binding is a separate gate;
tool metadata and the complete judge interface are part of the
instrument; selection leakage / proxy inversion / anti-signaling
need explicit treatment; no personal profile experiments in the
public repo.

## 12. What went wrong in this session

- Audit phase: read-only, correct, no thrashing. The audit doc
  was the right thing to write.
- Refactor phase: attempted, blocked by sandbox repeatedly
  dropping installed packages. Did not learn from the first
  failure; tried to re-install and re-test instead of stepping
  back. **Do not repeat.**
- Stopped when the user said "stop" mid-attempt. Good.

**Do not repeat**:
- Do not begin the refactor on the same turn as running the test suite.
- Do not retry the env install more than once.
- Do not re-audit from scratch — `METHODS_AUDIT.md` is the plan.
- Do not change priors or the public API silently.
- Do not introduce abstractions where deletions will do.

## 13. Recommended next-session plan

1. **Inspect** repo: `git status`, `git log --oneline -3`, confirm
   `main` clean. **Evidence required**: clean `main` + worktree
   on `feature/audit-and-handoff` with audit files.
2. **Install deps once**:
   ```
   pip install --break-system-packages --index-url https://pypi.org/simple/ \
     pymc pytensor arviz numpy scipy nutpie numpyro jax jaxlib pytest
   ```
3. **Run tests as observation**:
   `PYTHONPATH=src python3 -m pytest tests/ -q --tb=line`. Record
   pass/fail. **Do not change code in this step. Evidence required**:
   clean run output.
4. **Re-verify audit** findings against current source. **Evidence
   required**: `grep -n "pm.Potential\|pt.stack\|np.maximum"` in
   `btd.py` matches the audit's claims.
5. **Re-verify PyMC API** from current official docs. **Evidence
   required**: `pm.Categorical` accepts `logit_p` in the current
   version.
6. **Resolve the §9 STOP condition** with the user. **Evidence
   required**: explicit go-ahead on option 1 / 2 / 3.
7. **Run synthetic example** to establish a behavior baseline:
   `PYTHONPATH=src python3 examples/synthetic.py` and
   `examples/three_view.py`. **Evidence required**: reproducible
   ranking and position-effect sign.
8. **Replace BTD hand-rolled softmax** with `pm.Categorical(logit_p=...)`.
   Re-run tests + synthetic example. **Evidence required**: ranking
   unchanged, position-effect sign unchanged.
9. **Add shared `_davidson_log_probs` helper**; use in
   `summarize_btd` and `predict_btd`; replace `np.maximum` pattern
   with `scipy.special.softmax`. Re-run tests.
10. **Replace PPC** with `pm.sample_posterior_predictive`. Re-run
    tests.
11. **Vectorize** per-item / per-pair loops. Re-run tests.
12. **Add missing invariant tests** (audit brief items 17-19, 25-28).
    Re-run tests.
13. **External reference validation** (bpcs/Stan in dev env) if
    practical. If not, document that the canonical PyMC API is
    the substitute.
14. **Delete redundant code** (projected -273 LOC, -16%).
15. **Update docs** to match the final implementation; confirm
    `METHODS_AUDIT.md §G` parameterization crosswalk still holds.
16. **Commit only after all tests pass and synthetic example
    reproduces.** One commit, descriptive message.

## 14. Do-not-do list

The next agent must NOT:
- add features
- build a framework / plugin / registry / factory / provider
  abstraction / new base class
- write custom HMC / NUTS / leapfrog / mass-matrix / step-size /
  proposal / acceptance / chain-management code
- replace Davidson with plain Bradley-Terry
- add dependencies just because they exist
- compare θ magnitudes across unrelated fits
- treat reasoning traces as causal ground truth
- bless divergences using an arbitrary percentage threshold
- silently change priors
- silently break the public API
- rewrite working code before understanding its current semantics
- preserve custom math merely because it already exists
- perform broad refactors before establishing reference tests
- begin the refactor on the same turn as the test run
- retry `pm.Categorical(logit_p=...)` integration more than once
  if it fails; the API has been stable since v5.6.0
- touch the deprecated `fit` alias without a deprecation timeline
- introduce a new sampler backend without benchmarking

## 15. Commands for the next agent

```bash
# Enter repo
cd /run/csi/mount-root/nas/eab0d61a99b6696edb3d2aff87b585e8/pairwise-rank

# Inspect status / diff
git status --short
git diff --stat
git log --oneline -5

# Install deps (one shot, at the top of the session)
pip install --break-system-packages --index-url https://pypi.org/simple/ \
  pymc pytensor arviz numpy scipy nutpie numpyro jax jaxlib pytest

# Run tests (do not change code in this step)
PYTHONPATH=src python3 -m pytest tests/ -q --tb=line

# Inspect relevant files
cat pyproject.toml
cat src/pairwise_rank/__init__.py
head -100 src/pairwise_rank/btd.py
head -60 src/pairwise_rank/model.py
grep -n "pm.Potential\|pt.stack\|np.maximum" src/pairwise_rank/btd.py

# Read the audit
head -120 METHODS_AUDIT.md

# Synthetic baselines
PYTHONPATH=src python3 examples/synthetic.py
PYTHONPATH=src python3 examples/three_view.py

# After refactor: re-run synthetic and confirm ranking + position
# effect sign are unchanged.
```

## 16. Minimal context prompt for the fresh session

```
Read CONTEXT_HANDOFF.md completely before making changes.

You are continuing an audit-and-simplification pass on
`pairwise-rank`, a small Python library for Bayesian ordinal
paired-comparison ranking using PyMC + ArviZ. The previous
session delivered a read-only audit (METHODS_AUDIT.md) and
committed a rewrite of EXPERIMENT_DESIGN.md; the refactor was
deferred because the sandbox repeatedly dropped installed
dependencies.

State: `main` @ `ba9bcea` (clean). Worktree
`.worktrees/audit-handoff/` on `feature/audit-and-handoff` @
`befa8d3` (pushed) contains METHODS_AUDIT.md and CONTEXT_HANDOFF.md.

Job, in order:
1. Inspect the repo. Confirm git state and audit doc are intact.
2. Install dependencies once at the top
   (pymc, pytensor, arviz, numpy, scipy, nutpie, numpyro, jax,
   jaxlib, pytest).
3. Run the test suite to obtain a clean baseline pass/fail
   count. Record it. Do not change code in this step.
4. Resolve the STOP condition in CONTEXT_HANDOFF.md §9 with the
   user before any model change. The direct_summary tournament
   score denominator is wrong; the test pins the wrong value.
5. Re-verify the PyMC API claims in METHODS_AUDIT.md §B against
   current official docs.
6. Execute the refactor in the order in CONTEXT_HANDOFF.md §13.
   Do not reorder. Do not batch.
7. After each refactor step, re-run the test suite and the
   synthetic example. The recovered ranking and position-effect
   sign must be unchanged (within MC noise).
8. Use current official documentation to verify any API you
   are not certain about.
9. Preserve the exact statistical semantics in
   CONTEXT_HANDOFF.md §3. Especially: do not replace Davidson
   with plain Bradley-Terry; do not compare θ magnitudes across
   fits; do not bless divergences; do not treat reasoning
   traces as causal.
10. Prefer deletion and canonical library calls over new
    abstractions. Target: smaller, more boring codebase.
11. Do not add features, frameworks, plugins, registries, or
    new base classes. This project is a small tool, not a
    platform.
12. Show evidence before changing model mathematics. The
    exact Davidson parameterization is in CONTEXT_HANDOFF.md §8.
    The current code matches it. Any change must preserve it.
13. Commit only when all tests pass and the synthetic example
    reproduces. One commit, descriptive message.

Stop conditions (report to user before proceeding):
- Existing tests encode contradictory semantics (the
  tournament_score normalization — already known; resolve first).
- A proposed change would alter priors or model semantics.
- The PyMC API has changed since the docs read by the previous
  session.
- A reference validation reveals a real statistical discrepancy.
- Public API would need to break.
```

---

## Quality bar

This handoff distinguishes:

- **FACT**: verified from repository state, source code, pyproject,
  and `git` output in this session. Includes: HEAD commit, file
  LOC, public API list, test count, git state, recent commits.
- **DECISION**: methodological choices already made and committed
  to `EXPERIMENT_DESIGN.md` and the source. Includes: 3-level
  default, Davidson geometric-mean tie term, `pm.ZeroSumNormal`,
  `pm.OrderedLogistic`, position-neutral with `beta_right = 0`.
- **HYPOTHESIS**: suspected problems not yet verified end-to-end.
  The audit's findings about duplicated BTD math, hand-rolled
  softmax, hand-rolled PPC, divergence fallback to 0 are all
  HYPOTHESIS until the refactor runs and the synthetic example
  reproduces.
- **TODO**: work the next session must perform. The Phase 0 →
  Phase 15 list in `METHODS_AUDIT.md` and the §13 plan above.

The `tournament_score` [0, 2] bug is the only known case where
the repo (code + test) disagrees with the methodology (doc).
Resolve that first.
