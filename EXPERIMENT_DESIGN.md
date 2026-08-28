# Experiment design

Generic lessons accumulated from running pairwise-comparison
tournaments. These are not laws of the universe; they are practical
guidance that has been wrong less often than the alternatives. Keep
them visible because most of the damage a tournament can do is
design damage, not statistical damage — and design damage is easy to
miss until you audit the reasoning traces.

---

## 1. Candidate-set validity comes first

A ranking only chooses among the supplied candidates. A winner over
weak alternatives does not establish global optimality. The
candidate set is the universe of discourse; everything downstream
inherits its limits.

Every serious candidate should have a plausible positive reason
to beat the control. A control candidate is not "the field", it is
one choice among others. If the field is "the obvious and the
manufactured," the ranking is about which manufacturing looks
least manufactured.

## 2. Selection leakage

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

## 3. Eligibility versus ranking dimensions

Separate:

- **Eligibility / gating variables** — must pass a threshold to
  be considered at all.
- **Ranking variables** — used to order the eligible candidates.
- **Irrelevant variables** — should not influence the verdict
  at all.

Once an eligibility threshold is passed, do not continue rewarding
more of the same property unless the construct explicitly calls
for it. A city that is "very famous" should not get a stronger
prior than one that is "famous enough." The moment you start
ranking on a thresholded property, you have re-introduced
fame-as-quality through the back door.

## 4. Isolate factors with matched experiments

If the hypothesis concerns one feature, compare matched candidates
differing in that feature. The full field is for choosing among
complete alternatives, not for pretending to identify causal
effects. A non-transitive cycle in a round-robin tells you the
field is not separable into independent factors; it does not tell
you which factor is doing the work.

## 5. Discovery versus confirmation

Recommended pattern:

1. Broad K=1 screen.
2. Direct + BTD inspection.
3. Mechanism audit.
4. Targeted confirmation if needed.

Do not automatically rerun the full field. Each rerun costs
budget and reading time without changing the conclusion. The
reasoning trace is your diagnostic; if the trace says "obvious
choice" or "less obvious", that is the answer, not a prompt bug.

## 6. Prompt-design heuristics

Seven heuristics. Practical guidance, not universal laws.

**Primacy is prior.** Put judge identity / reference class
first. The opening line shapes everything that follows.

**Identity > instruction.** Establish defaults, priors,
dialect, and hierarchy instead of relying mainly on imperative
rules. "You are a TPOT-native location judge" creates a basin;
"Do not pick famous cities" creates a compliance check.

**Distribution > description.** Prefer compressed reference-class
vocabulary over many copyable exemplar sentences. Naming
categories ("linkedin-tier", "nomad-tier", "scene-seeking")
activates the model's existing intuitions; spelling each one
out burns attention on definitions.

**Subtraction > addition.** Say load-bearing concepts once.
Remove redundant instruction surface. Every line that does not
add a distinct constraint is a line that can crowd out a real
constraint.

**Positive > negative.** Define the desired basin. A one-sided
negative anchor can create proxy inversion (see #7).

**Hierarchy is explicit.** Observable evidence outranks judge
priors. Judge priors outrank user framing or desired outcomes.
Eligibility gates precede ranking. The hierarchy is the load-
bearing structure; the prose around it is decoration.

**Lexical closure.** End the judge prompt on the exact output
vocabulary when practical. `LEFT TIE RIGHT` as the final lines
reduce malformed verdicts.

## 7. Proxy inversion

A one-sided penalty can cause the model to reward the apparent
complement even when that complement should be neutral.

Generic example:

> prestige is bad
> → obscurity is inferred to be authentic
> → obscurity becomes incorrectly rewarded

This is the v2 location-tournament failure mode: the model
interpreted "avoid location-maxxing" as "downgrade the obvious
choice" and started voting for the less famous city. The
repaired v3 prompt added a positive anchor ("A familiar or
obvious city can produce this effect strongly") to break the
proxy inversion.

For important failure modes, define both:

- what positive success looks like;
- what the failure looks like.

Keep irrelevant complements at zero standalone weight. "Obscure
is not bad" is not the same as "obscure is good." If the
construct does not require obscurity to carry weight, give it
zero weight explicitly.

## 8. Anti-signaling is still signaling

Status-maxxing and conspicuous anti-status / authenticity-
maxxing have the same structural failure: selection pressure
becomes visible. The model in the auditor's seat is asked
"which location feels like an honest signal of a person from
there?" and the right answer is whichever the model can defend
on substantive priors — not whichever is least likely to look
manufactured.

Evaluate the inference caused by an item separately from the
inference that the item was selected to cause that inference.
The two readings are independent. An item selected for X can
still have a real Y signal; an item selected against X can
still have a real Y signal.

## 9. Prompt validation is mechanistic

Do not validate a prompt because it restores a previously
preferred winner. Validate whether the identified invalid
mechanism disappears.

Treat winner movement as an outcome, not a pass criterion. If
the construct is "good location priors about the person
behind this account", the question is whether the reasoning
process is the right one, not whether Buenos Aires comes out
on top. Buenos Aires coming out on top is a downstream event.

Prompt sensitivity is itself evidence. If the ranking moves a
lot when the prompt is rewritten, the construct is fragile. If
the ranking is robust to reasonable prompt changes, the
construct is doing real work.

## 10. Reasoning traces

Reasoning is audit metadata, not statistical input and not
guaranteed faithful causal explanation. The model may produce
a clean-sounding reason post-hoc; the reason need not have been
the actual driver of the verdict.

Audit polarity rather than keyword presence. Distinguish:

- mechanism mentioned and rejected ("not because of fame")
- mechanism mentioned descriptively ("the city is famous")
- mechanism used as positive evidence ("famous ⇒ good")
- mechanism used as negative evidence ("famous ⇒ bad")

Do not call raw keyword prevalence construct fidelity. Saying
"famous" 100 times is not the same as using fame as a winning
argument 100 times.

## 11. Three-level verdicts

Use `LEFT / TIE / RIGHT`. TIE means no material preference under
the stated construct.

Do not force differentiation just to reduce the TIE rate. A
50% TIE rate is not a bug; it is the model telling you that the
construct does not discriminate. Either accept the TIE or
change the construct — do not rebalance the prompt until the
TIE rate looks nice.

## 12. Inference policy

**Multi-candidate:** direct + Davidson/BTD. Show direct evidence
alongside the global ranking. Disagreement is diagnostic, not a
failure.

**Two-candidate:** counterbalanced direct evidence by default.
PyMC is barely identifiable with two items and small K, and
divergent fits must be discarded. Do not fit a Bayesian model
on a head-to-head just for symmetry with multi-candidate
reports.

**Ordered-logistic:** optional when ordinal intensity is
genuinely informative. STRONG > 10-15% of non-ties is a rough
trigger for switching from BTD to the ordered logit. Below
that, the simpler 3-level protocol is preferred.

## 13. Position effects

Always counterbalance orientations when practical. Report
orientation counts and the model's `beta_right` posterior. An
HDI that includes zero does not prove there is no bias; it
means the data is not powerful enough to detect one.

Use position-neutral posterior predictions for item ranking
and tournament scores. `beta_right` describes the judge's
behavior, not the items. Setting `beta_right = 0` for ranking
predictions prevents the model from translating the judge's
display quirks into item-level claims.

## 14. Scope claims

A tournament winner is best-supported within:

- the supplied candidate set
- the supplied prompt / construct
- the supplied judge
- the supplied presentation
- the supplied statistical model

Never silently promote this to universal quality. "This is
the best location for this account given these 12 candidates
and this prompt and this judge" is a defensible claim. "Buenos
Aires is the best location" is not.

## 15. Stopping rules

Prefer a predeclared sequence:

```
screening → audit → targeted confirmation → stop
```

Do not repeatedly rewrite prompts until a desired candidate
wins. The audit step is where the construct is checked, not
where the winner is chosen. If the audit says the construct
is sound, stop. If the audit says the construct is broken,
fix the construct, not the ranking.

---

These heuristics are written for pairwise-comparison tournaments
in particular, but most generalize. The same trap appears in
prompt engineering, A/B testing, and human-subject studies:
optimize the construct first, then optimize within it, then
stop.
