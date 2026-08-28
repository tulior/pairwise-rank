"""Three-view report: direct W/L/T, BTD ranking, M0 ordinal ranking.

Loads a list of observations, runs direct_summary, fit_btd, and fit_ordinal
on the same data, and produces a side-by-side table plus agreement
diagnostics. This is the routine report pattern for any
multi-candidate tournament (>=5 items, >=30 obs recommended).

For head-to-heads (2 items, <=30 obs) the PyMC fits are barely
identifiable; use direct_summary() alone in that case.

The function is pure: it returns a dict, writes nothing. The caller
decides what to do with the result (print, save to JSON, etc.).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .protocol import Observation
from .btd import fit_btd, summarize_btd, direct_summary, BTDFitResult
from .model import fit_ordinal, summarize as summarize_ordinal, FitResult


def three_view_report(
    observations: Iterable[Observation],
    *,
    item_ids: list[str] | None = None,
    draws: int = 2000,
    tune: int = 2500,
    chains: int = 4,
    target_accept: float = 0.99,
    seed: int = 42,
    hdi_prob: float = 0.9,
    include_ordinal: bool = True,
) -> dict:
    """Run direct_summary, fit_btd, and (optionally) fit_ordinal on
    the same observations and return a unified three-view report.

    include_ordinal defaults to True so the M0 cross-check is
    reported by default. Pass include_ordinal=False to skip the M0
    fit entirely; the report then only has direct and BTD.

    Returns a dict with keys:
        n_observations
        strong_collapsed: counts of LEFT_STRONG / RIGHT_STRONG collapsed by BTD
        direct:  per-item W/L/T and pairwise tallies (no model)
        btd:     BTDFitResult + summary dict
        m0:      FitResult + summary dict (only if include_ordinal=True)
        ranking: per-item table with W, L, T, BTD theta, BTD P(best),
                 M0 theta, M0 P(best), BTD rank, M0 rank, direct rank
        top1:    {direct, btd, m0 (or None), all_three_agree}
        theta_corr_btd_m0:    Pearson r between BTD and M0 theta means
                              (None if include_ordinal=False)
        pbest_corr_btd_m0:    Pearson r between BTD and M0 P(best) values
                              (None if include_ordinal=False)
    """
    obs = [o for o in observations if o.verdict]
    if not obs:
        raise ValueError("no completed observations to report on")

    # Infer item_ids if not provided
    if item_ids is None:
        seen = []
        seen_set = set()
        for o in obs:
            for i in (o.a, o.b, o.left, o.right):
                if i not in seen_set:
                    seen.append(i)
                    seen_set.add(i)
        item_ids = sorted(seen)

    # View 1: direct (no model)
    direct = direct_summary(obs)

    # View 2: BTD (default probabilistic model)
    btd_result: BTDFitResult = fit_btd(
        obs, item_ids=item_ids, draws=draws, tune=tune,
        chains=chains, target_accept=target_accept, seed=seed,
    )
    btd_summary = summarize_btd(btd_result, observations=obs, hdi_prob=hdi_prob)

    # View 3: M0 ordinal (optional cross-check)
    m0_result: FitResult | None = None
    m0_summary: dict | None = None
    m0_per: dict | None = None
    if include_ordinal:
        m0_result = fit_ordinal(
            obs, item_ids=item_ids, draws=draws, tune=tune,
            chains=chains, target_accept=target_accept, seed=seed,
        )
        m0_summary = summarize_ordinal(m0_result, observations=obs, hdi_prob=hdi_prob)
        m0_per = {row["id"]: row for row in m0_summary["per_item"]}

    btd_per = {row["id"]: row for row in btd_summary["per_item"]}

    def s_i(item_id: str) -> tuple[int, int, int, int]:
        w = direct["per_item"]["wins"].get(item_id, 0)
        l = direct["per_item"]["losses"].get(item_id, 0)
        t = direct["per_item"]["ties"].get(item_id, 0)
        return w - l, w, l, t

    # Per-item ranking table
    ranking = []
    for c in item_ids:
        net, w, l, t = s_i(c)
        row = {
            "id": c,
            "wins": w, "losses": l, "ties": t, "net": net,
            "btd_theta": btd_per[c]["theta_mean"],
            "btd_p_best": btd_per[c]["p_best"],
        }
        if m0_per is not None:
            row["m0_theta"] = m0_per[c]["theta_mean"]
            row["m0_p_best"] = m0_per[c]["p_best"]
        ranking.append(row)

    # Per-view ranks (1 = best)
    def assign_ranks_by(items: list, key, *, reverse=True) -> dict:
        sorted_items = sorted(items, key=lambda r: key(r), reverse=reverse)
        return {r["id"]: i + 1 for i, r in enumerate(sorted_items)}

    direct_ranks = assign_ranks_by(ranking, lambda r: r["net"])
    btd_ranks = assign_ranks_by(ranking, lambda r: r["btd_theta"])
    m0_ranks = assign_ranks_by(ranking, lambda r: r["m0_theta"]) if m0_per is not None else None
    for r in ranking:
        r["direct_rank"] = direct_ranks[r["id"]]
        r["btd_rank"] = btd_ranks[r["id"]]
        if m0_ranks is not None:
            r["m0_rank"] = m0_ranks[r["id"]]

    # Sort the ranking by direct rank, then btd rank
    ranking.sort(key=lambda r: (r["direct_rank"], r["btd_rank"]))

    # Top-1 agreement
    top1_direct = ranking[0]["id"]
    top1_btd = min(item_ids, key=lambda c: btd_ranks[c])
    if m0_ranks is not None:
        top1_m0 = min(item_ids, key=lambda c: m0_ranks[c])
        all_three_agree = (top1_direct == top1_btd == top1_m0)
    else:
        top1_m0 = None
        all_three_agree = (top1_direct == top1_btd)

    # Theta / P(best) correlations between BTD and M0
    if m0_per is not None:
        btd_thetas = np.array([btd_per[c]["theta_mean"] for c in item_ids])
        m0_thetas = np.array([m0_per[c]["theta_mean"] for c in item_ids])
        theta_corr = float(np.corrcoef(btd_thetas, m0_thetas)[0, 1])
        btd_pb = np.array([btd_per[c]["p_best"] for c in item_ids])
        m0_pb = np.array([m0_per[c]["p_best"] for c in item_ids])
        pbest_corr = float(np.corrcoef(btd_pb, m0_pb)[0, 1])
    else:
        theta_corr = None
        pbest_corr = None

    strong = btd_summary["config"]["strong_collapsed"]

    out = {
        "n_observations": len(obs),
        "n_items": len(item_ids),
        "item_ids": list(item_ids),
        "strong_collapsed": strong,
        "direct": direct,
        "btd_summary": btd_summary,
        "ranking": ranking,
        "top1": {
            "direct": top1_direct,
            "btd": top1_btd,
            "m0": top1_m0,
            "all_three_agree": all_three_agree,
        },
        "theta_corr_btd_m0": theta_corr,
        "pbest_corr_btd_m0": pbest_corr,
    }
    if m0_summary is not None:
        out["m0_summary"] = m0_summary
    return out


def print_three_view(report: dict, label: str = "") -> None:
    """Pretty-print a three_view_report() result."""
    out = report
    if label:
        print(f"\n{'='*78}\n# {label}\n{'='*78}")
    print(f"# n_observations = {out['n_observations']}, n_items = {out['n_items']}")
    sc = out["strong_collapsed"]
    print(f"# STRONG collapsed in BTD: {sc['total_collapsed']}/{out['n_observations']} "
          f"({sc['total_collapsed']/out['n_observations']*100:.1f}%)")

    has_m0 = out.get("m0_summary") is not None
    if has_m0:
        print(f"\n# {'rank':<5}{'item':<32}{'W':<4}{'L':<4}{'T':<4}"
              f"{'BTD θ':<10}{'BTD P(best)':<14}{'M0 θ':<10}{'M0 P(best)':<14}")
        print("-" * 100)
        for i, r in enumerate(out["ranking"], 1):
            item_label = r["id"] if r["id"] else "(empty)"
            print(f"  {i:<5}{item_label:<32}{r['wins']:<4}{r['losses']:<4}{r['ties']:<4}"
                  f"{r['btd_theta']:<+10.3f}{r['btd_p_best']:<14.3f}"
                  f"{r['m0_theta']:<+10.3f}{r['m0_p_best']:<14.3f}")
    else:
        print(f"\n# {'rank':<5}{'item':<32}{'W':<4}{'L':<4}{'T':<4}"
              f"{'BTD θ':<10}{'BTD P(best)':<14}")
        print("-" * 80)
        for i, r in enumerate(out["ranking"], 1):
            item_label = r["id"] if r["id"] else "(empty)"
            print(f"  {i:<5}{item_label:<32}{r['wins']:<4}{r['losses']:<4}{r['ties']:<4}"
                  f"{r['btd_theta']:<+10.3f}{r['btd_p_best']:<14.3f}")

    t1 = out["top1"]
    if has_m0:
        print(f"\n# Top-1: direct={t1['direct']!r}, btd={t1['btd']!r}, m0={t1['m0']!r}")
    else:
        print(f"\n# Top-1: direct={t1['direct']!r}, btd={t1['btd']!r}, m0=<skipped>")
    print(f"# All three agree: {'YES' if t1['all_three_agree'] else 'NO'}")
    if has_m0:
        print(f"# BTD vs M0 θ correlation: r = {out['theta_corr_btd_m0']:.4f}")
        print(f"# BTD vs M0 P(best) correlation: r = {out['pbest_corr_btd_m0']:.4f}")

    btd_pos = out["btd_summary"]["position_effect"]
    print(f"# Position effect (β_right, BTD): "
          f"{btd_pos['beta_right_mean']:+.3f} [{btd_pos['beta_right_hdi'][0]:+.3f}, {btd_pos['beta_right_hdi'][1]:+.3f}]")
    if has_m0:
        m0_pos = out["m0_summary"]["position_effect"]
        print(f"# Position effect (β_right, M0 ordinal): "
              f"{m0_pos['beta_right_mean']:+.3f} [{m0_pos['beta_right_hdi'][0]:+.3f}, {m0_pos['beta_right_hdi'][1]:+.3f}]")

    if "tie_parameter" in out["btd_summary"]:
        nu = out["btd_summary"]["tie_parameter"]
        print(f"# BTD tie parameter (ν): {nu['nu_mean']:.3f}  HDI90 [{nu['nu_hdi'][0]:.3f}, {nu['nu_hdi'][1]:.3f}]")
