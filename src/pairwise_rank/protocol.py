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
callable (left_id, right_id) -> Verdict. The package does not
provide a default prompt, an LLM tool schema, or a provider
abstraction. Those are the caller's job.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

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


JudgeFn = Callable[[str, str], str]  # (left_id, right_id) -> Verdict


@dataclass
class Observation:
    """One repeated judgment. Rows are never averaged.

    Fields:
      a, b: the canonical unordered pair in original candidate-list order
            (a appears before b in the input list).
      left, right: candidate ids in the displayed left and right slots.
      repeat: 1-based repeat index within the (a, b) cell.
      verdict: one of VERDICT_LEVELS, or empty string if not yet judged.
    """

    a: str
    b: str
    left: str
    right: str
    repeat: int
    verdict: str = ""


def observation_key(obs: Observation) -> tuple:
    """Dedup key. Stable across re-runs."""
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

    judge_fn is called once per unfinished row. If it returns a string
    not in VERDICT_LEVELS, ValueError is raised. If it raises an
    exception, the exception propagates. There is no built-in retry;
    wrap judge_fn with retry logic if needed.

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
        verdict = judge_fn(obs.left, obs.right)
        if verdict not in VERDICT_TO_CODE:
            raise ValueError(
                f"judge_fn returned invalid verdict: {verdict!r}; "
                f"expected one of {VERDICT_LEVELS}"
            )
        obs.verdict = verdict
        out.append(obs)
        done.add(observation_key(obs))
    return out


# ----------------------------------------------------------------------------
# JSON Lines persistence
# ----------------------------------------------------------------------------

def save_observations_jsonl(path: Path, observations: Iterable[Observation]) -> None:
    """Write observations as JSON Lines, one row per line."""
    with open(path, "w") as f:
        for obs in observations:
            f.write(json.dumps(asdict(obs)) + "\n")


def load_observations_jsonl(path: Path) -> list[Observation]:
    """Load observations from a JSON Lines file."""
    out: list[Observation] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(Observation(**d))
    return out
