"""pairwise-rank: small tools for reproducible pairwise ranking.

Architecture (v0.4):
    direct_summary   baseline / always (no model, raw W/L/T + tournament score)
    fit_btd          default probabilistic model (3-level Bradley-Terry-Davidson)
    fit_ordinal      optional / legacy (5-level ordered logit)
    fit              DEPRECATED alias for fit_ordinal

Verdict scale:
    VERDICT_LEVELS    default 3-level: LEFT, TIE, RIGHT
    VERDICT_LEVELS_5  5-level ordinal: LEFT_STRONG, LEFT, TIE, RIGHT, RIGHT_STRONG
    The default scale is 3-level because the 5-level ordinal
    information is rarely used in practice. STRONG verdicts
    occur in ~1-2% of non-ties, and BTD vs the ordered logit
    give r_theta > 0.99 and r_P(best) > 0.99 on multiple
    tournaments. Existing 5-level data on disk loads fine and
    is collapsed on use by BTD; no migration is required.

The collapse mapping is exposed as `collapse_to_3_level` and is
applied consistently in `direct_summary` and `fit_btd`.

Re-exports the public API. No side effects on import.
"""
from .protocol import (
    VERDICT_LEVELS,
    VERDICT_LEVELS_5,
    DEFAULT_VERDICT_LEVELS,
    Verdict,
    verdict_to_code,
    code_to_verdict,
    collapse_to_3_level,
    JudgeFn,
    Observation,
    observation_key,
    make_schedule,
    run_tournament,
    save_observations_jsonl,
    load_observations_jsonl,
)
from .model import fit, fit_ordinal, summarize, posterior_predictive_check, FitResult
from .btd import fit_btd, summarize_btd, direct_summary, BTDFitResult
from .report import three_view_report, print_three_view

__all__ = [
    # Verdict scale
    "VERDICT_LEVELS",
    "VERDICT_LEVELS_5",
    "DEFAULT_VERDICT_LEVELS",
    "Verdict",
    "verdict_to_code",
    "code_to_verdict",
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
    "fit_ordinal",        # optional 5-level ordered logit
    "fit",                # DEPRECATED alias for fit_ordinal
    "summarize",          # works for both models (signature-compatible)
    "summarize_btd",
    "posterior_predictive_check",
    "FitResult",
    "BTDFitResult",
    # Reports
    "direct_summary",     # baseline / always
    "three_view_report",  # direct + BTD + (optional) ordinal
    "print_three_view",
]
