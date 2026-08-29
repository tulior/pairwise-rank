# Experiment design

Generic lessons accumulated from running pairwise-comparison
tournaments. These are not laws of the universe; they are practical
guidance that has been wrong less often than the alternatives. Keep
them visible because most of the damage a tournament can do is
design damage, not statistical damage — and design damage is easy to
miss until you audit the reasoning traces.

A reader should be able to use this document to design, run, and
report a pairwise tournament without re-deriving the same failures.
The order is meant to be read once in sequence; later sections
assume the earlier ones.

---

## 1. What a pairwise experiment estimates

A pairwise tournament has three conditional layers. They are
distinct, and the right way to report a result depends on which
layer is in scope.

**Layer 1 — the construct.** What the tournament is trying to
measure. The field, the supported inference, the reference class,
the evidence pathway. See §2. The construct does not change with
sampler settings.

**Layer 2 — the judgment-generating instrument.** The complete
interface that turns a paired comparison into a verdict: the
candidate set (§3), the presentation (§4), the judge interface
including the function schema and any other effective instrument
surface (§5), the prompt (§6), and the judge / model itself.

**Layer 3 — the inferential model.** The statistical model that
turns per-cell verdicts into tournament-level summaries. Typically
direct counterbalanced evidence plus a Davidson / Bradley-Terry-
Davidson global model (§9). The model is conditional on its own
priors and likelihood assumptions.

A per-cell estimand is conditional on Layers 1 and 2:

```
Y_ij  ~  P(verdict | construct, candidates, presentation,
                 judge interface, judge)
```

A tournament-level estimand is conditional on all three:

```
P(θ, β, ν | Y, BTD assumptions)
```

A single paired comparison under the standard protocol therefore
estimates a posterior over LEFT / TIE / RIGHT — not over "the
true quality of A vs B" — and the tournament estimates
latent-strength contrasts under the global model, not absolute
quality, not deployment value, not anything outside the candidate
set, prompt, presentation, judge, instrument surface, or model
assumptions supplied.

A tournament does **not** estimate:

- absolute quality of any item;
- quality of an item outside the candidate set;
- the deployment value of the item (see §17);
- a property the judge had to imagine rather than read (see §3).

It is useful to write the per-cell estimand down before any prompt
work, and to keep it visible while the prompt and tool surface are
being designed. Most methodological failures become visible at this
level: the question drifts, the estimand silently changes, and the
posterior becomes evidence for a different claim than the one the
reader is told is being tested.

## 2. Define the construct before the candidates

The construct is the description of what the tournament is supposed
to be measuring. It is not a property of the candidate set, the
prompt, the tool, or the model. It is a property of the experiment.

A good construct specifies:

- **the field** — the population of artifacts being scored
  (a profile field, a text, an image, etc.);
- **the supported inference** — the kind of posterior the judge is
  being asked to update (e.g. "what kind of person is behind this
  account" rather than "which artifact is aesthetically superior");
- **the reference class** — the audience whose priors are being
  modeled;
- **the evidence pathway** — what visible features of the artifact
  are allowed to update which inferences.

The construct has to be specified well enough to predict the verdict
in obvious cases. If the construct cannot tell you which side of an
obvious comparison should win, the construct is not the thing the
tournament is actually measuring.

Construct drift is a common cause of an uninterpretable
result. The candidates and the prompt both drift, the construct
stays implicit, and the eventual winner is evidence for a claim
nobody stated.

## 3. Candidate eligibility and candidate-set validity

A ranking only chooses among the supplied candidates. The candidate
set is the universe of discourse; everything downstream inherits its
limits. A winner over weak alternatives does not establish global
optimality.

Before candidates enter the tournament, check:

- **Factual eligibility.** Does the candidate exist in the form
  supplied? A bio string, an image, a username — verify that what
  is being scored is what would actually be deployed.
- **Realizability.** A candidate that the account owner cannot or
  will not deploy is not a winner, even if it ranks first. A
  statistically dominant candidate that is undeployable has only
  ranked well against a counterfactual.
- **Evidence binding.** The candidate must be able to support the
  inferences the judge is asked to make about it. If the winning
  reasoning requires the artifact to *be* something it is not
  (a personal photograph, a self-authored text, a place the account
  owner has actually visited), see §17. False evidence binding is a
  deployment-validity failure, not an aesthetic one.
- **Threshold traits.** Properties that gate admission but should
  not be re-ranked once the threshold is passed. A field that is
  "famous enough" should not get a stronger prior than one that is
  "very famous." The moment you start ranking on a thresholded
  property, you have re-introduced the property through the back
  door.
- **Noisy-construct alternatives.** A control candidate is not
  "the field"; it is one choice among others. If the field is "the
  obvious and the manufactured," the ranking is about which
  manufacturing looks least manufactured.

## 4. Presentation fidelity and nuisance normalization

The presentation of a paired comparison is a nuisance variable if
it is not part of the construct. It becomes a bias if it is allowed
to update the verdict.

**Rule:** if a property is part of the deployment target, preserve
it. If it is merely unequal presentation during evaluation,
normalize it. Crop, lighting, resolution, aspect ratio, and the
like are not intrinsically nuisance variables — sometimes they
*are* the treatment. Do not normalize away the thing you are trying
to measure.

Sources of presentation noise:

- **Position effects.** The judge may prefer the LEFT or RIGHT slot
  for reasons unrelated to the artifact. Always counterbalance
  orientations when practical, and report the model's `beta_right`
  posterior. An HDI that includes zero does not prove there is no
  bias; it means the data is not powerful enough to detect one.
- **Order of presentation across multiple items.** If the same
  judge is shown multiple comparisons in sequence, earlier
  comparisons may shift anchors that influence later ones. Within a
  single tournament this is usually negligible; across long
  multi-stage campaigns it is worth checking.
- **Image detail, crop, and aspect ratio.** For image comparisons,
  the detail level, the framing crop, and the aspect ratio are
  presentation variables. Normalize them across candidates before
  the tournament. A 3:1 banner crop and a square crop are different
  visual fields, and a difference in outcome may be the crop, not
  the photograph.
- **Resolution, lighting, color profile.** Same idea, different
  variable. A photograph presented at low resolution has less
  detail to score; a photograph presented with strong color
  grading carries a presentation cue that an ungraded version
  would not.
- **Order of the candidates within the prompt.** A candidate
  described first in the prompt receives more attention than one
  described second. If you describe a strategy, drill into it, and
  describe its sibling, the order can affect which is judged more
  carefully.

The position-neutral summary is the right report for item ranking
and tournament score. `beta_right` describes the judge's behavior,
not the items. Setting `beta_right = 0` for ranking predictions
prevents the model from translating the judge's display quirks
into item-level claims.

## 5. The judge interface is part of the instrument

When the judge is asked to commit its answer via a function call,
the function schema sits immediately around the act of commitment.
The judge interface — the complete instrument surface — includes
all of the following. Some are part of the function schema; others
are not literally "prompt" but are part of the experimental
treatment. Changing any of them is a prompt change.

- the function **name**;
- the function **description**;
- the parameter **name**, **description**, and **enum**;
- the **output vocabulary** the model is asked to commit to;
- the **message structure** — the order and grouping of the input
  parts, the framing, the role labels, the section delimiters;
- the **response shape** and any structured-output controls;
- the `tool_choice` setting;
- the **reasoning setting** (if any);
- the **model and provider configuration**;
- the **rendering configuration** — image detail, text rendering,
  asset formatting, output truncation, anything that changes how
  the input is presented to the model.

These are not implementation details; they are part of the
effective rubric. The model commits its answer in the language of
the interface, not the language of the prompt body.

Concretely, the difference between these two schemas is not
cosmetic:

```
name: record_preference
description: "Record the pairwise bio preference as a 3-level
             ordinal vote."

name: record_posterior_comparison
description: "Record which bio produces the better TPOT-native
             posterior about the person behind the account. Use
             TIE when neither bio produces a materially better
             posterior."
```

The first quietly reframes the task from poster inference into
subjective preference, and "ordinal vote" adds irrelevant
statistical language. The second restates the actual estimand and
closes the semantics of TIE.

Three rules for the schema:

1. **Name the function as a passive recording, not an active
   choice.** The model has already done the judging; the tool merely
   records the result. `record_X` keeps the prompt and the tool in
   the same conceptual basin. Active verbs (`choose_better_bio`,
   `select_best_profile`, `rank_bios`, `evaluate_profile`) subtly
   push toward decisiveness and against TIE.
2. **Name the parameter after the decision variable, not the storage
   form.** `verdict` is fine; `ordinal_vote`, `left_wins`,
   `bio_score`, `winner` are not. The model sees the parameter name
   when committing the answer.
3. **Make TIE a first-class option in the description.** Many models
   default to picking a side unless the schema tells them TIE is
   normal. Phrases like "Use TIE when neither X produces a
   materially better Y" raise the TIE rate legitimately without
   changing the question.

Tempting names to avoid:

| name | failure mode |
|---|---|
| `record_preference` | too aesthetic / subjective |
| `record_X_preference` | frames "which X do I like" |
| `choose_better_X` | suppresses TIE |
| `select_best_X` | implies a winner must exist |
| `record_ordinal_vote` | statistical implementation leaks into judge task |
| `evaluate_X` | too broad |
| `record_tpot_fit` | turns target into scene-fit, not person posterior |
| `record_authenticity` | proxy capture |

The lexical path at the end of the prompt should match the function
schema:

```
construct phrase
    ↓
function name
    ↓
LEFT / TIE / RIGHT
```

The final few tokens of the prompt, the function name, and the
function description should be inside the same conceptual basin. A
clean function name makes the prompt's last line shorter and more
direct, and a direct last line makes the function name easier to
choose.

The documented `tool_choice` for the Responses API is `"none"` or
`"auto"`, not a named-function-forcing object. With a single tool
defined and the prompt explicitly asking the model to call it,
`"auto"` normally produces the call, but this is not an API-level
guarantee. The runner must enforce the protocol client-side: a
completed response without exactly one valid function call is a
malformed judgment, retry it. Do not convert free text into a
verdict.

This whole surface is part of the experimental treatment.
Changing the function name, the function description, the
parameter names, the parameter descriptions, the enum, the
output vocabulary, the message structure, the reasoning setting,
the model/provider configuration, the rendering configuration, or
any other interface variable is a prompt change. Run-to-run
attribution only holds when all of these are held constant.

## 6. Prompt design and proxy failure modes

The prompt is not the whole instrument — see §5 for the function
schema. But it is the load-bearing part of the treatment.

Seven heuristics, in roughly the order they matter:

1. **Primacy is prior.** Put judge identity and reference class
   first. The opening line shapes everything that follows. A
   construct that appears on line 30 of a long prompt is a construct
   the model is not anchoring on.
2. **Identity > instruction.** Establish defaults, priors, dialect,
   and hierarchy instead of relying mainly on imperative rules. "You
   are a TPOT-native field judge" creates a basin; "Do not pick the
   famous option" creates a compliance check. Compliance-mode
   prompts are more brittle to paraphrase and more vulnerable to
   proxy inversion.
3. **Distribution > description.** Prefer compressed reference-class
   vocabulary over many copyable exemplar sentences. Naming a
   category (e.g. "scene-seeking", "linkedin-tier", "nomad-tier")
   activates the model's existing intuitions; spelling each one out
   burns attention on definitions.
4. **Subtraction > addition.** Say load-bearing concepts once. Every
   line that does not add a distinct constraint is a line that can
   crowd out a real constraint. Additive rubric bloat degrades
   everything in the rubric.
5. **Positive > negative.** Define the desired basin. A one-sided
   negative anchor can create proxy inversion (see below).
6. **Hierarchy is explicit.** Observable evidence outranks judge
   priors. Judge priors outrank user framing or desired outcomes.
   Eligibility gates precede ranking. The hierarchy is the load-
   bearing structure; the prose around it is decoration.
7. **Lexical closure.** End the judge prompt on the exact output
   vocabulary when practical. `LEFT TIE RIGHT` as the final lines
   reduce malformed verdicts and give the model one last anchor on
   the verdict space.

**Proxy inversion.** A one-sided penalty can cause the model to
reward the apparent complement even when that complement should be
neutral. Generic example:

> prestige is bad
> → obscurity is inferred to be authentic
> → obscurity becomes incorrectly rewarded

For important failure modes, define both:

- what positive success looks like;
- what the failure looks like.

Keep irrelevant complements at zero standalone weight. "Obscure is
not bad" is not the same as "obscure is good." If the construct does
not require obscurity to carry weight, give it zero weight
explicitly.

**Anti-signaling is still signaling.** Status-maxxing and
conspicuous anti-status / authenticity-maxxing have the same
structural failure: selection pressure becomes visible. The right
answer is whichever the model can defend on substantive priors, not
whichever is least likely to look manufactured. (See §15 for the
search-side version of this failure.)

**Prompt validation is mechanistic, not winner-driven.** Do not
validate a prompt because it restores a previously preferred
winner. Validate whether the identified invalid mechanism
disappears. Treat winner movement as an outcome, not a pass
criterion. Prompt sensitivity is itself evidence: if the ranking
moves a lot when the prompt is rewritten, the construct is fragile;
if the ranking is robust to reasonable prompt changes, the
construct is doing real work.

## 7. Balanced pairwise protocol

The protocol has three structural choices that should be settled
before the first call goes out:

- **Three-outcome verdicts.** Use `LEFT / TIE / RIGHT`. TIE means
  no material preference under the stated construct. Do not force
  differentiation to reduce the TIE rate. A high TIE rate is the
  model telling you the construct does not discriminate, not a bug
  to be balanced away. Either accept the TIE or change the
  construct.
- **Both orientations.** For every (a, b) pair, run (a as LEFT, b
  as RIGHT) and (b as LEFT, a as RIGHT). The orientation count
  feeds the position-effect estimate and prevents a single
  presentation order from dominating the verdict. See §4.
- **K = number of repeats per cell.** K=1 is appropriate for
  discovery screens. K≥3 is appropriate for confirmation. K is not
  a knob for "tightening" the result; it is the unit of statistical
  evidence at the per-cell level.

The protocol and the model are independent. A clean protocol on a
broken construct still produces a broken result. A clean construct
on a broken protocol still produces a broken result. Both have to
be right.

## 8. Direct evidence and global inference

For multi-candidate tournaments, report both:

- **Direct counterbalanced evidence.** Wins, losses, ties, and a
  tie-adjusted tournament score per item. For a complete balanced
  tournament with both orientations and K repeats, the per-item
  probability-like score is

  ```
  S_i^direct = (W_i + 0.5 * T_i) / (2 * K * (N - 1))
  ```

  range [0, 1], position-neutral (does not depend on which slot
  the item appeared in). The denominator is `2 * K * (N - 1)`
  because each item faces the other `N - 1` items in `K` repeats
  on each side, and the maximum possible `(W + 0.5 * T)` is
  `2 * K * (N - 1)`. Divide by `K * (N - 1)` instead and the
  statistic lives on [0, 2] and is a different quantity. Watch
  the denominator.
- **A global latent-strength model** (typically BTD — see §9).

Disagreement between direct and global is strain, not automatically
a failure. A non-transitive cycle in direct evidence is real if it
involves three or more candidates; with two candidates, an
orientation-asymmetric verdict is a position effect, not a cycle.
Three or more candidates with a real cycle indicates the
latent-strength assumption is wrong for this field.

For two-candidate comparisons, the right report is the
counterbalanced direct evidence. Fitting a Bayesian model on a
head-to-head is rarely worth it; with two items and small K the
posterior is barely identifiable, divergent fits have to be
discarded, and the per-cell counts already say what needs to be
said. The BTD on a two-candidate tournament is a presentation
choice, not a statistical necessity.

## 9. Model choice: direct, Davidson/BTD, ordinal models

The model choice is driven by the measurement, not by what
statistics happen to be available.

```
two-candidate matched question
-> counterbalanced direct tally

multi-candidate LEFT/TIE/RIGHT tournament
-> direct summaries + Davidson / Bradley-Terry-Davidson global model

genuinely ordinal intensity data
-> ordered-logistic model
```

Three model classes, with their actual roles:

- **Direct counterbalanced tally.** No model. Wins, losses, ties,
  tie-adjusted score. Always report. For a two-candidate
  contrast, this is the primary report; a Bayesian model is
  rarely worth fitting.
- **Davidson / Bradley-Terry-Davidson (Davidson 1970).** Latent
  item strength λ_i, position-bias parameter ν, with the tie
  probability proportional to `ν · sqrt(λ_i · λ_j)` — the
  geometric-mean Davidson form, **not** an arbitrary 3-outcome
  Bradley-Terry extension. The default global model for
  multi-candidate LEFT/TIE/RIGHT tournaments. Reports θ, P(best),
  P(top-k), expected rank, and β_right (the position effect, on
  a log-odds scale).
- **Ordered logistic / M0.** Treats an ordinal 5-level verdict
  scale as the response. Use **only** when the ordinal intensity
  is genuinely part of the measurement — when the difference
  between, say, "slight right preference" and "strong right
  preference" carries information the BTD's TIE category cannot
  represent. In a 3-level LEFT/TIE/RIGHT protocol, M0 is not the
  right tool. The 5-level protocol and the M0 model are part of
  the same legacy path; the default today is BTD on the 3-level
  verdicts.

Choosing among them is a measurement question, not a statistical
completeness question. "We have ordinal verdicts and a fancy model
that uses them" is not a reason to prefer M0. The question is
whether the ordinal level carries information the BTD's TIE
category is failing to capture.

For position-effect reporting, the relevant quantity is `beta_right`
on the log-odds scale. A positive value means the judge tends to
prefer the RIGHT slot independent of item strength. Use position-
neutral posterior predictions (set `beta_right = 0` for the
ranking summary) so that display-side behavior does not show up as
item-level claims.

## 10. Diagnostics and fit failures

A successful fit is not just "0 divergences." Diagnostics are about
the geometry of the posterior, not the absence of crashes.

- **Any divergence is a fit warning and a geometry failure
  requiring investigation.** A divergent transition is the sampler
  telling you the posterior has a region it could not navigate.
  There is no "acceptable" divergence fraction. A non-zero rate
  requires investigation regardless of how small. Increasing K or
  `target_accept` does not repair geometry; it papers over it. If
  the geometry is broken, the fix is in the model or the data, not
  in sampler settings.
- **R-hat close to 1 is necessary, not sufficient.** Four chains
  agreeing at R-hat = 1.0 says the chains mixed to the same
  distribution. It does not say that distribution is the right
  one for the data. Inspect the per-parameter posteriors, not just
  the headline R-hat.
- **ESS bulk and ESS tail are Monte Carlo efficiency
  diagnostics.** They estimate roughly how many independent draws
  the autocorrelated chain is worth for estimating a quantity. They
  are **not** measures of statistical coverage. Bulk ESS matters
  for central posterior summaries (means, medians). Tail ESS
  matters for quantiles and tail-region behavior (HDI endpoints,
  P(best), expected rank). Low ESS in either metric means the
  corresponding summary is high-variance; a chain with R-hat = 1.0
  and ESS = 50 has not explored enough for the kinds of summaries
  the report will quote.
- **Posterior predictive checks (PPC).** A nice PPC on one chosen
  statistic does not establish calibration. A single summary
  statistic can pass while the predictive distribution is wrong in
  other ways. Use multiple statistics; check the joint predictive
  behavior; check the residuals the chosen statistic hides.
- **β_right HDI including zero.** This is "not enough data to
  detect a bias," not "no bias." With small K, a real position
  effect can be invisible. Always counterbalance; report the
  estimate and its interval; do not let the HDI test stand in for
  the design.
- **TIE rate.** Diagnostic, not a quality score. A high TIE rate
  may mean the construct does not discriminate; it may also mean
  the candidates are genuinely equivalent on the construct. A low
  TIE rate means the judge differentiated the presented
  alternatives; it does not mean the experiment was intrinsically
  good. Low TIE rate is consistent with the judge being
  overconfident, with the construct being over-narrow, or with the
  candidates being designed to force a side.

## 11. Reasoning traces and post-hoc mechanism audits

Reasoning is audit metadata, not statistical input and not
guaranteed faithful causal explanation. The model may produce a
clean-sounding reason post-hoc; the reason need not have been the
actual driver of the verdict. Treat the trace as a hypothesis
about why the verdict happened, not as evidence that the verdict
happened for that reason.

When auditing traces:

- **Audit polarity, not keyword presence.** Distinguish:
  - mechanism mentioned and rejected ("not because of fame");
  - mechanism mentioned descriptively ("the city is famous");
  - mechanism used as positive evidence ("famous ⇒ good");
  - mechanism used as negative evidence ("famous ⇒ bad").

  Saying "famous" 100 times is not the same as using fame as a
  winning argument 100 times.

- **Record polarity and effect on the verdict.** A mechanism
  mentioned positively is different from one mentioned
  descriptively. Record both: the polarity, and whether the trace
  shows the mechanism actually weighted the verdict.

- **The audit taxonomy must not leak into the judge request.**
  If the judge is told, mid-task, that a particular mechanism is
  being audited, the audit becomes a prompt intervention. Save the
  reasoning, run the audit offline, and keep the audit
  vocabulary separate from the judge prompt.

- **Mention is not endorsement.** A reasoning trace that says "I
  could see this being a famous-tier location, but..." is using
  fame as a *negative* signal, not a positive one. A trace that
  says "this is a famous location, so..." is using it positively.
  Same keyword, different role.

## 12. Matched ablation for component effects

An omnibus ranking tells you whether a composite candidate works.
It does not tell you which component of the candidate caused the
result. A composite may succeed despite a harmful component rather
than because of it.

Suppose two candidates differ in two primitives:

```
A + X
B + Y
```

If `B + Y` wins, the result does not identify an effect for `B` or
`Y`. Possible explanations include:

- `B` helps and `Y` helps;
- `B` helps while `Y` hurts;
- `Y` helps while `B` hurts;
- an interaction between `B` and `Y` matters;
- neither isolated component reproduces the omnibus result.

To estimate the effect of a primitive, hold the surrounding context
fixed:

```
A + X  vs  B + X
A + Y  vs  B + Y
```

When practical, repeat the matched contrast across more than one
surrounding context. This distinguishes a stable primitive effect
from a context-specific interaction.

For two-candidate matched contrasts, counterbalanced direct
evidence is the primary report. A global ranking model is usually
unnecessary for a two-candidate contrast.

**Rule:** composite rankings answer "does this complete alternative
work?" Matched ablations answer "what changes when this one
primitive changes?" Credit a primitive only when a comparison that
isolates that primitive supports the claim.

A useful workflow:

```
omnibus screen
-> identify composite frontier
-> decompose candidates into primitives
-> matched contrasts
-> post-hoc mechanism audit (on saved reasoning, see §11)
```

Do not infer component effects by subtracting latent scores from an
omnibus tournament. Those scores are global, field-relative
quantities and generally confound all differences between the
candidates.

## 13. Discovery, confirmation, and stopping

The right shape of a tournament campaign is a sequence, not a loop.

```
screening (K=1, broad)
-> audit (mechanism check on saved reasoning)
-> targeted confirmation (K>=3 on the close pairs)
-> stop
```

The screening step is cheap and identifies the composite frontier.
The audit step is where the construct is checked, not where the
winner is chosen. The confirmation step resolves the close pairs
that actually need more evidence. The stop is a real step: there
should be a named end state and a reason to be there.

Common failure modes:

- **Reopening settled components without new decision-relevant
  uncertainty.** Once a component is locked, reopening it costs
  budget and reading time without changing the conclusion. Curiosity
  is not a reason to re-run. The threshold for reopening is: there
  is a specific downstream decision that depends on a specific
  number changing.
- **Treating confirmation as more screens.** A K>=3 confirmation
  run is not a re-do of the screen; it is a focused direct test of
  the close decision. Do not expand the candidate set during
  confirmation; do not change the prompt; do not add a new
  category.
- **Optimizing the prompt until a desired candidate wins.** If the
  audit says the construct is sound, stop. If the audit says the
  construct is broken, fix the construct, not the ranking.
  Repeatedly rewriting the prompt until the desired winner emerges
  is selection leakage on the prompt — see §15.

## 14. Coarse-to-fine search and abstraction collapse

When a tournament has both coarse stages (compare strategies by
description) and fine stages (compare rendered artifacts), they
are not equivalent.

Coarse stages answer: "which strategy produces a better posterior
under imagined execution?" The judge necessarily imagines a
representative realization. The dominant strategy tends to absorb a
category-level abstraction advantage — the parent "transit" can
sound good because the judge imagines the best member of the
parent.

Fine stages answer: "which artifact produces a better posterior as
encountered?" The judge cannot abstract away the actual image.
Incidental properties of the specific artifact (crop, weather,
signage, country, time of day) become ineliminable part of the
comparison.

These are different questions. The coarse stage is a category
experiment. The fine stage is an artifact-selection experiment.
The transition between them is delicate.

The semantic hierarchy has three levels of inference, and they do
not transfer upward:

1. **Strategy-level inference** — what the judge said about
   categories or subcategories described in the abstract.
2. **Micro-type inference** — what the judge said about a
   specific subgenre within a category.
3. **Artifact inference** — what the judge said about a specific
   rendered artifact.

A fine-stage artifact win does not imply the micro-type is best.
A coarse-stage micro-type win does not imply the parent strategy
is best. A strategy-level win does not imply the strategy is the
right banner strategy at all — see §3 for the candidate-set
constraint, §15 for selection leakage, and §17 for deployment
validity. The discipline is to write the conclusion at the
stage's level: an artifact conclusion, a micro-type conclusion,
and a strategy conclusion are three different claims and need
three different pieces of evidence.

Two failure modes in particular:

1. **Retain the abstract parent in the fine stage.** A parent
   category that won the coarse stage can continue to "sound good"
   at the fine stage because the judge imagines a particular member
   rather than evaluating a specific artifact. The parent
   effectively still has the abstraction advantage.

   The fix is to retain a *concrete sibling incumbent* from the
   strongest alternative branch, not the abstract parent. The
   sibling is a specific real artifact the judge cannot re-imagine.

2. **Treat a fine-stage winner as a category claim.** "Image X
   won" is not "category Y is the best motif." The semantic
   hierarchy gets you to a useful image search distribution. The
   fine stage then selects among those artifacts. Image-level and
   category-level inferences do not transfer; the next category in
   the same logical role may select a different image for
   incidental reasons (lighting, weather, the way a particular
   photographer cropped). Do not back-propagate an image win into
   a categorical causal claim. See §16.

A useful workflow:

```
coarse screen (descriptions, K=1)
-> identify composite frontier
-> drill within winning branch
   with sibling incumbent from strongest rival branch
-> lockdown (concrete artifacts, K>=3)
-> image selection is not a category experiment; stop
```

At the fine stage, also keep `<NO FIELD>` (or its equivalent
control) alive as long as it has not been clearly dominated.
Decomposition sanity checks whether the parent is genuinely
preferred or merely sounds preferred against thin alternatives.

In fine stages, incidentally contaminated artifacts — a photograph
with a rainbow that pulls "pastoral," a facade that reads as
"council estate," an image with loud commercial signage — are not
"merely imperfect realizations." The fine stage is artifact
selection. Incidental properties of the artifact are the
experiment. Accept them, swap them, or drop the slot. Do not
average them out.

## 15. Selection leakage and anti-signaling

Selection leakage occurs when the search procedure optimizes a
property whose value depends on appearing unoptimized or
organically acquired.

Generic examples:

- Optimizing an "effortless" phrase until it looks manufactured.
- Selecting an "authentic" signal specifically because it scores
  as authentic.
- Optimizing anti-status signaling until anti-status itself
  becomes the signal.

Core rule: do not endlessly optimize for appearing unoptimized.

When candidates are effectively tied after crossing an
acceptability threshold, genuine preference, an exogenous fact, or
random selection may be more construct-faithful than another
ranking pass.

Status-maxxing and conspicuous anti-status / authenticity-maxxing
have the same structural failure: selection pressure becomes
visible. The model in the auditor's seat is asked "which field
option feels like an honest signal of a person from there?" and
the right answer is whichever the model can defend on substantive
priors — not whichever is least likely to look manufactured.

Evaluate the inference caused by an item separately from the
inference that the item was selected to cause that inference. The
two readings are independent. An item selected for X can still
have a real Y signal; an item selected against X can still have a
real Y signal.

Discovery and confirmation both need stopping rules. The
selection-leakage failure is not "we ranked too aggressively" — it
is "we keep ranking without ever crossing a stopping threshold."
If the audit says the construct is sound, stop. If the construct
is broken, fix the construct. Reopening to chase a less-leaked
winner is itself a leak.

## 16. Artifact-level versus strategy-level inference

When the experiment transitions from comparing strategies by
description to comparing rendered artifacts, the estimand changes.
A winning artifact is evidence for that artifact among those
candidates, not retrospective proof that the strategy or
micro-type from which it was drawn is globally optimal.

Concretely:

- "Image X won" is evidence that image X produced the better
  posterior among the supplied candidates. It is not evidence that
  its micro-type is the best motif.
- "The transit subcategory won at the coarse stage" is evidence
  about how the judge scores the *description* of a transit
  subcategory. It is not a forecast for which specific
  implementation of transit will win at the fine stage.
- "Field A tends to rank above field B in this campaign" is
  evidence within the candidate set, the prompt, the judge, the
  presentation, and the model. A different candidate set, a
  different prompt, or a different judge may produce a different
  ranking.

The temptation to generalize is strong because the rule is
mechanically simple: a property of a tournament is not a property
of the world. The discipline is to write conclusions at the scope
the experiment actually supports. "This artifact is the best
choice in this campaign" is defensible. "This artifact type is
the right one" is not.

## 17. Deployment validity and false evidence binding

A tournament winner is not automatically deployable. The question
"would this artifact still be desirable if its true provenance
were disclosed?" is a separate question from "did this artifact
rank first?"

False evidence binding is a deployment-validity failure: the
artifact cannot actually support the inference that drove its
win. Examples:

- A banner image that wins because the judge infers the account
  owner photographed the scene, when the image is in fact a
  publicly sourced photograph.
- A bio line that wins because the judge infers the account owner
  wrote it from a real experience, when the line is in fact a
  constructed joke.
- A username that wins because the judge infers it is a real
  handle on another platform, when it is in fact a fresh
  registration.
- A display name that wins because the judge infers it is a real
  legal name, when it is in fact a constructed handle.

**When to suspect false evidence binding.** The trigger is not
the reasoning trace alone — reasoning is post-hoc metadata, not
privileged causal access (see §11). The trigger is one of the
following:

- **Known provenance conflict.** The artifact's actual provenance
  (publicly sourced, constructed, fake, fresh registration, etc.)
  conflicts with an inference the judge is likely to draw
  naturally from the artifact. This is a pre-experiment check,
  done before running the tournament using the candidate-set
  properties — not the reasoning.
- **Reasoning audit signal.** A post-hoc audit of saved reasoning
  shows the winning argument chain crediting the artifact for
  unsupported provenance, authorship, ownership, or firsthand
  experience. The audit surfaces the hypothesis; it does not
  decide it.

When the trigger fires, the response is a **targeted matched
rerun, not another screen**: rerun only the decisive comparison,
with true provenance disclosed in the judge prompt. This rerun
supplies the actual evidence for or against deployment. If the
same winner survives the provenance correction, the win is real
and the mechanism is now a legitimate one (e.g. "choice of
ordinary, unposed urban observation" rather than "this person
has personally visited the depicted location"). If the other side
wins after provenance is neutralized, the deployment decision
changes.

Do not preemptively inject provenance disclaimers when they are
not needed. A provenance paragraph is itself part of the judge
interface (§5) and changes the natural-viewing estimand. It should
be added when the audit shows false evidence binding, not as a
default safety blanket.

## 18. Reporting, scope, and claim hygiene

A tournament winner is best-supported within:

- the supplied candidate set;
- the supplied prompt / construct;
- the supplied judge / model;
- the supplied judge interface — function schema, message
  structure, output vocabulary, reasoning setting, model/provider
  configuration, rendering configuration, and any other effective
  instrument surface (see §5);
- the supplied presentation (crop, aspect, detail, order, position;
  see §4);
- the supplied statistical model.

Never silently promote this to universal quality. The headline
"X is the best" is rarely defensible; "X is the best-supported
within the candidate set, prompt, judge, presentation, and model
tested" is.

Claim hygiene for the quantities the model reports:

- **θ is field-relative, and θ magnitudes from different fits are
  not comparable.** A θ value is a position in the latent space
  of one tournament. The prior and the candidate set both shift
  the scale; the units are not absolute. Two items from different
  fits cannot be ranked against each other by their θ values.
  Compare items within a single fit, never across fits. "Same θ
  in two fits" is a coincidence of relative position, not a
  portable measurement.
- **P(best) is a joint posterior event.** "P(best) = 0.7" means
  "in 70% of posterior samples, this item is the maximum-θ item."
  It is not a frequentist probability that the item is best in the
  world, and it is not comparable across tournaments with
  different candidate sets.
- **Direct-vs-global disagreement is strain, not failure.** It
  indicates the global model is not a perfect fit, or the data
  has noise the model has smoothed over, or the field has
  structure the latent-strength representation does not capture.
  Show both reports; let the reader see the strain. Strain is
  not a failure, but it is also not a license to pick whichever
  report says what you wanted.
- **A non-transitive cycle in a round-robin needs three or more
  candidates.** Two candidates with an orientation-asymmetric
  verdict is a position effect, not a cycle. Three or more
  candidates with a real cycle indicates the latent-strength
  assumption is wrong for this field.
- **TIE rate is diagnostic, not a quality score.** See §10.
- **β_right HDI including zero is "not enough data to detect a
  bias," not "no bias."** See §4 and §10. Conversely, an HDI
  that excludes zero does not mean the position-neutral ranking
  is trustworthy; if the underlying fit is broken (geometry
  failures, low ESS, R-hat > 1.0), the precision of the
  exclusion is also broken. Trust the diagnostics first, the
  interval second.

The strongest single rule: every winner is conditional on
candidate set, judge, prompt/tool surface, presentation, protocol,
and statistical model. Conclusions are written at exactly that
scope. Anything looser is a different claim than the experiment
supported.

## 19. Experimental checklist

A pre-flight, in-flight, and post-flight summary. Use it as a
spine; the sections above give the rationale for each item.

**Before any call goes out:**

- The construct is written down, including the field, the
  supported inference, the reference class, and the evidence
  pathway. (§2)
- The candidate set has been checked for factual eligibility,
  realizability, evidence binding, and threshold traits. (§3)
- The presentation has been normalized (orientation plan, image
  detail / crop / aspect, order). (§4)
- The judge interface (prompt + function name + function
  description + parameter schema + `tool_choice`) is part of the
  same revision; any change to one is logged. (§5, §6)
- The protocol decisions are settled: 3-level verdicts, both
  orientations, K value, and the model class. (§7, §9)

**During the run:**

- Divergence rate, R-hat, ESS bulk, ESS tail are monitored. A
  non-zero divergence rate is a signal, not a number to bless. (§10)
- The `beta_right` posterior is tracked. An HDI that includes zero
  does not mean "no bias." (§4, §10)
- TIE rate is monitored as a diagnostic, not optimized. (§7, §10)

**After the run, before reporting:**

- A mechanism audit on the saved reasoning, with polarity
  recorded. Mention ≠ endorsement. The audit taxonomy is not in
  the judge request. (§11)
- Composite winners are not promoted to component claims. Primitives
  that drive the result are identified by matched ablations. (§12)
- For multi-candidate tournaments, both direct counterbalanced
  evidence and the global model are reported. Disagreement is
  shown. (§8)
- For two-candidate confirmations, the global model is optional;
  direct counterbalanced evidence is the primary report. (§8, §9)
- Audit: is the construct doing real work, or has the prompt
  drifted? Is the winner robust to reasonable prompt changes?
  (§6, §13)
- Audit: does the winning reasoning require the artifact to *be*
  something it is not? If yes, rerun the decisive comparison with
  true provenance disclosed. (§17)
- Headline claims are written at the scope the experiment
  actually supports. θ is not compared across fits. P(best) is
  presented as a joint posterior event. β_right is reported with
  its interval. (§18)
- The campaign has a stopping rule, and the rule has been
  honored. The next reopening requires a named downstream
  decision, not curiosity. (§13, §15)

---

These heuristics are written for pairwise-comparison tournaments
in particular, but most generalize. The same trap appears in
prompt engineering, A/B testing, and human-subject studies:
optimize the construct first, optimize within it, then stop.
