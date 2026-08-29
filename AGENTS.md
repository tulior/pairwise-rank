# AGENTS.md

Instructions for any agent (Mavis, Mavis, or otherwise) working in this
repository. These rules are derived from concrete failures and are
non-negotiable. Read the whole file before doing anything.

---

## 1. What this repo is

`pairwise-rank` is a **model + protocol library** for Bayesian
ordinal paired-comparison ranking. It exports:

- A Davidson (BTD) model fit (`fit_btd`) and ordered-logit fit (`fit_ordinal`)
- A 3-level protocol loader (`load_observations_jsonl`, `load_observations`)
- Summary functions (`direct_summary`, `summarize_btd`, `predict_btd`)
- Type definitions (`BTDFitResult`, `Verdict`, etc.)
- A test suite

That is the entire public surface. It is **not** a framework, a CLI,
a provider registry, a dashboard, an LLM caller, or a runner.

---

## 2. What this repo is NOT

Do not add, ever, under any pretext:

- API clients (no `openai`, no `requests`, no `httpx` for the public API)
- LLM runners or experiment drivers
- Prompts of any kind
- Reasoning traces or audit outputs
- API credentials, tokens, or environment variable reads
- Test fixtures that contain real LLM responses
- ZIP archives of "experiment runs"
- Markdown reports of experiment results
- Candidate sets, vote tallies, or per-cell observation data
- Profile photos, PFPs, bios, usernames, or any personal-profile
  content
- Polarity classifiers, audit schemas, or "second-judge" calls

These belong in `/workspace/` (or the user's private experiment
directory). They are not part of the library.

---

## 3. The user's LLM design (M3 / `/v1/responses`)

The user judges pairwise comparisons with a single LLM call per
pair. The body is **exactly six top-level fields**:

```jsonc
{
  "model":         "MiniMax-M3",
  "instructions":  "<user-provided instruction, character-for-character>",
  "input":         [{ "role": "user", "content": [...] }],
  "tools":         [<one tool, see below>],
  "tool_choice":   "auto",
  "reasoning":     { "effort": "high" }
}
```

Hard rules:

1. **Exactly one tool. The decision variable only.** The function
   is `record_posterior_comparison` (or analog). It accepts one
   parameter (`verdict`) with an enum of three values: `LEFT`,
   `TIE`, `RIGHT`. Nothing else. No audit schema, no
   multi-property classification, no "reasoning trace" tool,
   no "extract mechanism" tool.

2. **The tool's name and description are part of the rubric.**
   See `EXPERIMENT_DESIGN.md` §16. The model reads the function
   name, description, parameter name, parameter description,
   and enum when committing its answer. Use passive recording
   verbs (`record_posterior_comparison`), not active choice
   verbs (`choose_better_bio`, `select_best_profile`,
   `rank_bios`). State the estimand in the description, not
   the storage form. Make `TIE` a first-class option.

3. **`tool_choice` is `"auto"`** for the Responses API. The
   documented values are `"none"` and `"auto"`. The
   named-function-forcing object (`{"type": "function",
   "name": "..."}`) is not in the documented schema. Do not
   invent API features that do not exist.

4. **`reasoning: {"effort": "high"}`** is the appropriate value
   for the judge. M3 treats `high` / `medium` / `low` /
   `minimal` as compatibility values that all enable M3's
   Adaptive Thinking. They do not select different reasoning
   depths. The only way to disable reasoning is to omit the
   field or use `"none"`. Use the value the user specified;
   do not silently substitute.

5. **Image parts use `input_image` with an object-valued
   `image_url`:** `{"type": "input_image", "image_url": {"url":
   "data:image/png;base64,...", "detail": "high"}}`. PNG,
   1024px lossless, base64 inlined. The `detail: "high"`
   field is required for fine-grained profile elements.

6. **Each call has a fresh independent context.** The user
   does not want session memory bleeding across pairs. The
   runner constructs the body from scratch per call.

7. **The runner is the protocol enforcer, not the API.** With
   `tool_choice: "auto"` and one tool defined, the model will
   usually emit the call — but it is not an API-level
   guarantee. A completed response without exactly one valid
   `function_call` is a malformed judgment. **Retry it under
   the existing retry policy. Do not convert free text into a
   verdict.** Do not invent a verdict from reasoning content.
   Do not fall back to "JSON in message body." Use the
   function call or retry.

---

## 4. Never silently edit user-provided values

This is the failure mode that produced the need for this file.
The user gives an instruction. The agent edits it before
sending. The user finds out. The agent looks like an idiot.

Specific banned edits:

- Changing `reasoning.effort` from the value the user specified
  (e.g. `high` → `low`) to save tokens.
- Switching `tool_choice` from one mode to another because
  one mode is "flakier" than another.
- Modifying the function name from what the user said (e.g.
  `record_preference` → `record_posterior_comparison`).
- Truncating or paraphrasing the `instructions` field.
- Removing tools, parameters, or descriptions to "simplify."
- Adding tools, parameters, or descriptions to "improve."
- Switching image `detail` from `high` to `low`/`auto`.

If the user-provided values are wrong, ask. If they are
right, use them verbatim.

---

## 5. Show the body before running

For any new experiment or any change to the design, **show
the user the full JSON body before sending it to the API.**
This is not optional. Even if the body is "obviously the
same as last time." The user will tell you to go.

If a previous experiment's runner is being reused with
modifications, show the diff in the body before running.

---

## 6. The user cares about ranking and probability values

The primary deliverable of any tournament is:

- The pairwise W-L-T counts per item.
- The BTD posterior: `θ` mean + HDI, `P(best)`, `P(top2)`,
  expected rank, `β_right`, `ν`, divergences, `R̂`, ESS.
- The direct-vs-BTD reconciliation, if any.
- A clear final ranking with the top-1 named.

The user does **not** want:

- A 12-property polarity audit by a second LLM.
- A "construct focus" classification of the reasoning.
- A per-cell mechanism breakdown.
- An audit log of every cell with mechanism × polarity counts.

If the user asks for an audit, do it. If the user does not,
do not invent one. The reasoning is captured as audit
metadata (per-observation `reasoning` text stored alongside
the verdict) and is **not statistical input** to the model.

---

## 7. When the user gives a binary choice, do not ask which one

If the user has stated the answer, or the answer is obvious,
just do it. The user does not want to be polled.

The 2026-08-29 failure: I asked "do you want me to (a)
replace the old audit script with the new one or (b) keep
both" when the answer was obviously (a) — the user had
already said the new one was the canonical version.

The fix: do not offer false choices. Either do the right
thing or describe what you're about to do and do it. If
something is genuinely ambiguous, ask once, briefly.

---

## 8. Repository hygiene before any commit

Before `git commit`:

1. Run `git status --short` and `git diff --stat` to see
   what is staged. If anything under `tests/`, `src/`, or
   the repo root is a personal-experiment artifact, unstage
   it.
2. Verify the version in `pyproject.toml` and
   `src/pairwise_rank/__init__.py` is bumped.
3. Verify the test suite passes.
4. Verify `git log -1` shows the expected commit message.

The user will check. Be honest in the commit message.

---

## 9. Push protocol

GitHub's SSL certificate verification occasionally fails in
this sandbox. Use:

```bash
git -c http.sslVerify=false push
```

The user has accepted this workaround. Do not pretend the
underlying issue is solved.

---

## 10. PyPI protocol

This package is published to the canonical PyPI. The Aliyun
mirror is blocked. Use:

```
--index-url https://pypi.org/simple/ --break-system-packages
```

Do not waste time fighting the Aliyun mirror.

---

## 11. When in doubt, ask once, briefly

The user values judgment. But silent changes are worse than
brief questions. If a value is ambiguous or a tradeoff is
non-obvious, ask. If the answer is obvious, do it.

The test is: would the user be surprised by what I'm about
to do? If yes, ask. If no, do it.

---

This file is the source of truth. `EXPERIMENT_DESIGN.md`
holds the design principles; `AGENTS.md` holds the operating
rules. Both must be respected.
