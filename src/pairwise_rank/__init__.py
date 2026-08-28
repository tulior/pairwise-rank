"""pairwise_rank: small research tool for repeated balanced pairwise ordinal judgments.

Re-exports the public API. No side effects on import.
"""
from .protocol import (
    VERDICT_LEVELS,
    Verdict,
    verdict_to_code,
    code_to_verdict,
    JudgeFn,
    Observation,
    observation_key,
    make_schedule,
    run_tournament,
    save_observations_jsonl,
    load_observations_jsonl,
)
from .model import fit, summarize, posterior_predictive_check, FitResult
from .btd import fit_btd, summarize_btd, direct_summary, BTDFitResult

__all__ = [
    "VERDICT_LEVELS",
    "Verdict",
    "verdict_to_code",
    "code_to_verdict",
    "JudgeFn",
    "Observation",
    "observation_key",
    "make_schedule",
    "run_tournament",
    "save_observations_jsonl",
    "load_observations_jsonl",
    "fit",
    "summarize",
    "posterior_predictive_check",
    "FitResult",
    "fit_btd",
    "summarize_btd",
    "direct_summary",
    "BTDFitResult",
]
