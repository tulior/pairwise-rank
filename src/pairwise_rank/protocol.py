"""Tournament protocol: schedule, run, persist.

Owns:
  Verdict (alias for str, restricted to VERDICT_LEVELS)
  verdict_to_code / code_to_verdict
  Observation
  observation_key
  make_schedule(candidate_ids, repeats)
  run_tournament(candidate_ids, judge_fn, repeats=3, existing=())
  save_observations_jsonl / load_observations_jsonl

The protocol is construct-agnostic. The judge is a caller-supplied
callable. A simple judge returns a Verdict string. A judge that wants
to also record audit metadata (e.g. model reasoning text) can return
a (Verdict, str) tuple; the second element is stored on the
Observation as `reasoning` and is ignored by the ranking model.

The package does not provide a default prompt, an LLM tool schema, or
a provider abstraction. Those are the caller's job.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Callable, Iterable, Tuple, Union

VERDICT_LEVELS: tuple[str, ...] = (
    "LEFT_STRONG",
    "LEFT",
    "TIE",
    "RIGHT",
    "RIGHT_STRONG",
)

# A Verdict is one of the five ordinal labels.
Verdict = str

VERDICT_TO_CODE: dict[str, int] = {v: i for i, v in enumerate(VERDICT_LEVELS)}


def verdict_to_code(verdict: str) -> int:
    if verdict not in VERDICT_TO_CODE:
        raise ValueError(f"unknown verdict: {verdict!r}; expected one of {VERDICT_LEVELS}")
    return VERDICT_TO_CODE[verdict]


def code_to_verdict(code: int) -> str:
    if not 0 <= code <= 4:
        raise ValueError(f"verdict code out of range: {code}")
    return VERDICT_LEVELS[code]


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
) -> list[Observation]:
    """Execute the full tournament schedule, skipping already-completed keys.

    Returns a list containing the existing observations (verbatim) plus
    the newly completed ones. The order of the returned list is:
    first the existing observations in their original order, then the
    new ones in schedule order.

    judge_fn is called once per unfinished row. It may return either a
    Verdict string or a (Verdict, reasoning_str) tuple. The reasoning
    string is stored on the Observation but is not used by the model.
    If the returned verdict is not in VERDICT_LEVELS, ValueError is
    raised. If judge_fn raises an exception, the exception propagates.
    There is no built-in retry; wrap judge_fn with retry logic if
    needed.

    The function makes no assumptions about the judge, the prompt, the
    modality, or the persistence layer. It only handles scheduling,
    dedup, and verdict validation.
    """
    schedule = make_schedule(candidate_ids, repeats)
    done = {observation_key(o) for o in existing}
    out = list(existing)
    for obs in schedule:
        if observation_key(obs) in done:
            continue
        result = judge_fn(obs.left, obs.right)
        verdict, reasoning = _split_judge_return(result)
        if verdict not in VERDICT_TO_CODE:
            raise ValueError(
                f"judge_fn returned invalid verdict: {verdict!r}; "
                f"expected one of {VERDICT_LEVELS}"
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

