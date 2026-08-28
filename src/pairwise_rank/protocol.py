"""Tournament protocol: schedule, run, persist.

Owns:
  VERDICT_LEVELS (default 3-level: LEFT, TIE, RIGHT)
  VERDICT_LEVELS_5 (5-level ordinal: LEFT_STRONG, LEFT, TIE, RIGHT, RIGHT_STRONG)
  DEFAULT_VERDICT_LEVELS (alias for VERDICT_LEVELS)
  verdict_to_code / code_to_verdict / collapse_to_3_level
  Observation
  observation_key
  make_schedule(candidate_ids, repeats)
  run_tournament(candidate_ids, judge_fn, repeats=3, existing=(), verdict_levels=VERDICT_LEVELS)
  save_observations_jsonl / load_observations_jsonl

The protocol is construct-agnostic. The judge is a caller-supplied
callable. A simple judge returns a Verdict string. A judge that wants
to also record audit metadata (e.g. model reasoning text) can return
a (Verdict, str) tuple; the second element is stored on the
Observation as `reasoning` and is ignored by the ranking model.

Verdict scale:
  The default is the 3-level scale (LEFT, TIE, RIGHT). The 5-level
  scale is available as VERDICT_LEVELS_5 for backward compatibility
  and for prompts that genuinely elicit intensity information. New
  code should use the 3-level default unless there is a specific
  reason to capture STRONG.

Backward compatibility:
  Existing observations on disk with 5-level verdicts (LEFT_STRONG,
  RIGHT_STRONG) load correctly. fit_btd and direct_summary collapse
  STRONG into ordinary LEFT/RIGHT internally, so old data works
  with the new default without any migration step.

The package does not provide a default prompt, an LLM tool schema, or
a provider abstraction. Those are the caller's job. The canonical
3-level tool schema can be obtained from the example in
`examples/three_view.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

VERDICT_LEVELS_5: tuple[str, ...] = (
    "LEFT_STRONG",
    "LEFT",
    "TIE",
    "RIGHT",
    "RIGHT_STRONG",
)

# Default 3-level verdict scale. The accumulated evidence across
# multiple tournaments (textual, bio, prefix-match, hex batch, 0x
# class) shows that STRONG verdicts occur in <= 2% of observations
# and the collapsed 3-level inference is essentially identical to
# the 5-level inference (r_theta > 0.99, r_P(best) > 0.99). The
# simpler protocol also reduces tool-schema friction and the rate
# of malformed verdicts.
#
# The 5-level scale is preserved for backward compatibility and
# for the cases where intensity genuinely matters (see model.py:
# fit_ordinal). New code should use the 3-level default.
VERDICT_LEVELS: tuple[str, ...] = (
    "LEFT",
    "TIE",
    "RIGHT",
)

# Alias for clarity in new code: prefer this over VERDICT_LEVELS
# when reading.
DEFAULT_VERDICT_LEVELS: tuple[str, ...] = VERDICT_LEVELS

# A Verdict is one of the labels in whichever scale the caller is
# using. By default it is one of the three-level labels.
Verdict = str

# Built from the 5-level scale so legacy data loads correctly and
# so _split_judge_return accepts any of the 5 codes. Validation in
# run_tournament uses verdict_levels=VERDICT_LEVELS by default,
# which only accepts the 3-level codes; pass verdict_levels=
# VERDICT_LEVELS_5 to accept the 5-level scale.
VERDICT_TO_CODE: dict[str, int] = {v: i for i, v in enumerate(VERDICT_LEVELS_5)}

# Code mapping for 3-level scale (used by BTD).
VERDICT_TO_CODE_3: dict[str, int] = {v: i for i, v in enumerate(VERDICT_LEVELS)}


def verdict_to_code(verdict: str) -> int:
    """Map a 5-level verdict to its code (0..4).

    For backward compatibility, also accepts 3-level verdicts and
    returns the corresponding code. Use _btd_code() for the 3-level
    BTD likelihood mapping (left wins / tie / right wins).
    """
    if verdict in VERDICT_TO_CODE:
        return VERDICT_TO_CODE[verdict]
    if verdict in VERDICT_TO_CODE_3:
        return VERDICT_TO_CODE_3[verdict]
    raise ValueError(
        f"unknown verdict: {verdict!r}; expected one of {VERDICT_LEVELS_5}"
    )


def code_to_verdict(code: int) -> str:
    if not 0 <= code <= 4:
        raise ValueError(f"verdict code out of range: {code}")
    return VERDICT_LEVELS_5[code]


def collapse_to_3_level(verdict: str) -> str:
    """Collapse a 5-level verdict to 3-level: STRONG -> ordinary.

    LEFT_STRONG  -> LEFT
    RIGHT_STRONG -> RIGHT
    TIE / LEFT / RIGHT pass through unchanged.

    This is what BTD and direct_summary do internally.
    """
    if verdict == "LEFT_STRONG":
        return "LEFT"
    if verdict == "RIGHT_STRONG":
        return "RIGHT"
    return verdict


# A JudgeFn may return either:
#   - a Verdict string (just the verdict)
#   - a (Verdict, str) tuple (verdict and free-form audit metadata,
#     e.g. the model's reasoning text). The second element is stored
#     on the Observation as `reasoning` and is not used by the model.
JudgeReturn = Union[str, Tuple[str, str]]
JudgeFn = Callable[[str, str], JudgeReturn]


def _split_judge_return(result: JudgeReturn) -> Tuple[str, str]:
    """Return (verdict, reasoning) from a judge_fn return value."""
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError(
                f"judge_fn returned tuple of length {len(result)}; expected 2 (verdict, reasoning)"
            )
        verdict, reasoning = result
        return verdict, "" if reasoning is None else str(reasoning)
    return result, ""


@dataclass
class Observation:
    """One repeated judgment. Rows are never averaged.

    Fields:
      a, b: the canonical unordered pair in original candidate-list order
            (a appears before b in the input list).
      left, right: candidate ids in the displayed left and right slots.
      repeat: 1-based repeat index within the (a, b) cell.
      verdict: one of VERDICT_LEVELS, or empty string if not yet judged.
      reasoning: free-form audit metadata (e.g. the model's reasoning
                 text). Stored alongside the verdict but ignored by the
                 ranking model. Default empty string for backward
                 compatibility with rows written before this field
                 existed.
    """

    a: str
    b: str
    left: str
    right: str
    repeat: int
    verdict: str = ""
    reasoning: str = ""


def observation_key(obs: Observation) -> tuple:
    """Dedup key. Stable across re-runs.

    Reasoning is audit metadata and is intentionally not part of the
    key. Two observations that agree on (a, b, left, right, repeat)
    are the same row even if their reasoning text differs.
    """
    return (obs.a, obs.b, obs.left, obs.right, obs.repeat)


def make_schedule(candidate_ids: list[str], repeats: int) -> list[Observation]:
    """Build a deterministic schedule: every unordered pair, both
    orientations, K repeats each. Verdicts are empty placeholders.

    The canonical pair (a, b) keeps the order from the input list:
    a appears at a smaller index than b.

    For each pair we produce two orientations:
      - left = a, right = b   (L_first, a on the left)
      - left = b, right = a   (R_first, b on the left)

    Each orientation has K repeats numbered 1..K.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    out: list[Observation] = []
    n = len(candidate_ids)
    for i in range(n):
        for j in range(i + 1, n):
            a = candidate_ids[i]
            b = candidate_ids[j]
            for left, right in ((a, b), (b, a)):
                for r in range(1, repeats + 1):
                    out.append(Observation(
                        a=a, b=b, left=left, right=right, repeat=r, verdict="",
                    ))
    return out


def run_tournament(
    candidate_ids: list[str],
    judge_fn: JudgeFn,
    repeats: int = 3,
    existing: Iterable[Observation] = (),
    verdict_levels: tuple[str, ...] = VERDICT_LEVELS,
) -> list[Observation]:
    """Execute the full tournament schedule, skipping already-completed keys.

    Returns a list containing the existing observations (verbatim) plus
    the newly completed ones. The order of the returned list is:
    first the existing observations in their original order, then the
    new ones in schedule order.

    judge_fn is called once per unfinished row. It may return either a
    Verdict string or a (Verdict, reasoning_str) tuple. The reasoning
    string is stored on the Observation but is not used by the model.
    If the returned verdict is not in verdict_levels, ValueError is
    raised. If judge_fn raises an exception, the exception propagates.
    There is no built-in retry; wrap judge_fn with retry logic if
    needed.

    verdict_levels defaults to the 3-level scale (LEFT, TIE, RIGHT).
    Pass VERDICT_LEVELS_5 to use the 5-level ordinal scale (LEFT_STRONG,
    LEFT, TIE, RIGHT, RIGHT_STRONG). The 3-level scale is recommended
    for new code: the accumulated evidence across many tournaments
    shows STRONG verdicts in <= 2% of observations and the 3-level
    inference is essentially identical to the 5-level inference.

    The function makes no assumptions about the judge, the prompt, the
    modality, or the persistence layer. It only handles scheduling,
    dedup, and verdict validation.
    """
    allowed = set(verdict_levels)
    schedule = make_schedule(candidate_ids, repeats)
    done = {observation_key(o) for o in existing}
    out = list(existing)
    for obs in schedule:
        if observation_key(obs) in done:
            continue
        result = judge_fn(obs.left, obs.right)
        verdict, reasoning = _split_judge_return(result)
        if verdict not in allowed:
            raise ValueError(
                f"judge_fn returned invalid verdict: {verdict!r}; "
                f"expected one of {list(verdict_levels)}"
            )
        obs.verdict = verdict
        obs.reasoning = reasoning
        out.append(obs)
        done.add(observation_key(obs))
    return out


# ----------------------------------------------------------------------------
# JSON Lines persistence
# ----------------------------------------------------------------------------

# Names of fields that the on-disk row format may omit for backward
# compatibility. Each missing field is backfilled with its dataclass
# default.
_OPTIONAL_FIELDS = {"reasoning"}


def save_observations_jsonl(path: Path, observations: Iterable[Observation]) -> None:
    """Write observations as JSON Lines, one row per line."""
    with open(path, "w") as f:
        for obs in observations:
            f.write(json.dumps(asdict(obs)) + "\n")


def load_observations_jsonl(path: Path) -> list[Observation]:
    """Load observations from a JSON Lines file.

    Rows written before the `reasoning` field was added (or any
    future optional field) load with the dataclass default for the
    missing key. Unknown keys in the row are ignored.
    """
    valid_fields = {f.name for f in fields(Observation)}
    out: list[Observation] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Backfill any optional field the row omits.
            for k in _OPTIONAL_FIELDS:
                d.setdefault(k, "")
            # Drop keys the dataclass doesn't know about, so adding
            # new fields in the future doesn't break old loaders.
            d = {k: v for k, v in d.items() if k in valid_fields}
            out.append(Observation(**d))
    return out

