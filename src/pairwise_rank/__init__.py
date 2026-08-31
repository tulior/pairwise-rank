"""pairwise-rank: small tools for reproducible pairwise ranking.

Architecture (v0.5):
    direct_summary   baseline / always (no model, raw W/L/T + tournament score)
    fit_btd          default probabilistic model (3-level Bradley-Terry-Davidson)

The only supported methodology is the 3-level scale (LEFT, TIE, RIGHT)
with Davidson / BTD as the global inference model. Legacy 5-level
verdicts (LEFT_STRONG, RIGHT_STRONG) are accepted as input but
collapsed to ordinary wins/losses on ingest -- no 5-level inference
is performed. Use VERDICT_LEVELS_5 with run_tournament to accept
legacy verdict strings; fit_btd and direct_summary handle the
collapse internally.

Re-exports the public API. No side effects on import.
"""
from .protocol import (
    VERDICT_LEVELS,
    VERDICT_LEVELS_5,
    DEFAULT_VERDICT_LEVELS,
    Verdict,
    collapse_to_3_level,
    JudgeFn,
    Observation,
    observation_key,
    make_schedule,
    run_tournament,
    save_observations_jsonl,
    load_observations_jsonl,
)
from .btd import fit_btd, summarize_btd, direct_summary, predict_btd, BTDFitResult

__all__ = [
    # Verdict scale
    "VERDICT_LEVELS",
    "VERDICT_LEVELS_5",
    "DEFAULT_VERDICT_LEVELS",
    "Verdict",
    "collapse_to_3_level",
    # Protocol
    "JudgeFn",
    "Observation",
    "observation_key",
    "make_schedule",
    "run_tournament",
    "save_observations_jsonl",
    "load_observations_jsonl",
    # Models
    "fit_btd",            # default probabilistic model
    "summarize_btd",
    "predict_btd",        # per-cell (orientation-aware) BTD likelihood
    "BTDFitResult",
    # Reports
    "direct_summary",     # baseline / always
]
