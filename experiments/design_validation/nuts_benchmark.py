"""NUTS fitting benchmark for the existing UNMODIFIED ``fit_btd``.

Measures wall time, divergences, R-hat, and bulk ESS at
N in {32, 100, 300, 1000} using a sparse degree-6 connectivity
that mimics the bootstrap of the adaptive design layer.

The benchmark is observational only. It does NOT change
inference backend, priors, or model. If N=1000 is too slow for
one fit, time the first ~100s and report partial results; do
not switch to VI, MAP, Laplace, or another model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "model_falsification", "scripts"))

from dgp import dgp_transitive  # noqa: E402

from pairwise_rank import Observation  # noqa: E402
from pairwise_rank.btd import fit_btd, summarize_btd  # noqa: E402
from pairwise_rank.design import make_sparse_bootstrap  # noqa: E402


VERDICT_FROM_CODE = {0: "LEFT", 1: "TIE", 2: "RIGHT"}


@dataclass
class BenchmarkResult:
    n_items: int
    n_unordered_pairs: int
    draws: int
    tune: int
    chains: int
    target_accept: float
    wall_time_s: float
    divergences: int
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    success: bool
    error: str | None = None


def build_sparse_observations(tournament, item_ids: list[str], degree: int = 6) -> list[Observation]:
    pairs = make_sparse_bootstrap(item_ids, degree=degree, seed=0)
    by_oriented: dict[tuple[str, str], str] = {}
    for li, ri, vc in zip(tournament.left, tournament.right, tournament.verdict):
        by_oriented[(item_ids[int(li)], item_ids[int(ri)])] = VERDICT_FROM_CODE[int(vc)]
    observations: list[Observation] = []
    for a, b in pairs:
        for left, right in ((a, b), (b, a)):
            verdict = by_oriented.get((left, right), "TIE")
            observations.append(Observation(
                a=a, b=b, left=left, right=right, repeat=1, verdict=verdict,
            ))
    return observations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-list", default="32,100,300,1000")
    parser.add_argument("--out", default="results/nuts_results.json")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--tune", type=int, default=300)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--max-wall-s", type=int, default=600,
                        help="abort a single fit after this many seconds")
    args = parser.parse_args(argv)
    n_list = [int(s) for s in args.n_list.split(",")]
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    results: list[BenchmarkResult] = []
    for n in n_list:
        print(f"[N={n}] generating DGP", flush=True)
        item_ids = [f"item_{k}" for k in range(n)]
        t = dgp_transitive(n=n, K=1, seed=0,
                           theta=np.linspace(1.5, -1.5, n))
        obs = build_sparse_observations(t, item_ids, degree=args.degree)
        n_pairs = len({(o.a, o.b) if o.a < o.b else (o.b, o.a) for o in obs})
        print(f"  sparse pairs={n_pairs}; obs={len(obs)}", flush=True)
        t0 = time.time()
        try:
            fit = fit_btd(obs, item_ids=item_ids, seed=0,
                          draws=args.draws, tune=args.tune,
                          chains=args.chains,
                          target_accept=args.target_accept)
            elapsed = time.time() - t0
            summary = summarize_btd(fit, obs, position_neutral=True)
            diag = summary.get("sampler_diagnostics", {})
            divergences = int(diag.get("divergences", 0))
            max_rhat = float(diag.get("max_rhat", float("nan")))
            min_bulk = float(diag.get("min_bulk_ess", float("nan")))
            min_tail = float(diag.get("min_tail_ess", float("nan")))
            success = True
            err = None
            print(f"  N={n} fit done in {elapsed:.1f}s; "
                  f"div={divergences} rhat={max_rhat:.3f} "
                  f"min_ess_bulk={min_bulk:.0f} min_ess_tail={min_tail:.0f}",
                  flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            divergences = 0
            max_rhat = float("nan")
            min_bulk = float("nan")
            min_tail = float("nan")
            success = False
            err = f"{type(e).__name__}: {e}"
            print(f"  N={n} FAILED after {elapsed:.1f}s: {err}", flush=True)
        results.append(BenchmarkResult(
            n_items=n, n_unordered_pairs=n_pairs,
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept,
            wall_time_s=elapsed, divergences=divergences,
            max_rhat=max_rhat, min_bulk_ess=min_bulk, min_tail_ess=min_tail,
            success=success, error=err,
        ))
        if elapsed > args.max_wall_s and not success:
            print(f"  N={n} exceeded max-wall-s, aborting further N",
                  flush=True)
            break

    out = {
        "draws": args.draws, "tune": args.tune, "chains": args.chains,
        "degree": args.degree, "results": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
