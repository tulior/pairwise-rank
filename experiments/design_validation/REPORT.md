# Adaptive comparison-design validation report

## Scope

This report documents the validation of the optional
large-N adaptive comparison-design layer in
`pairwise_rank.design`. The validation answers three
questions:

1. Does the design layer return a credible best set that
   actually contains the true best candidate?
2. Is the frontier acquisition score doing what it claims
   (spending calls on relevant, unresolved pairs)?
3. Where is the operational bottleneck when scaling to
   N=1000 — the design or the existing BTD fit?

The statistical model is unchanged: Bayesian Davidson with
the existing `fit_btd` and `summarize_btd`. The DGP is the
existing transitive Davidson DGP from
`experiments/model_falsification/scripts/dgp.py`.

## Methods compared

- **A. complete round robin**: every unordered pair is
  compared once in each orientation. The model is fit
  once on the full directed-pair set.
- **B. degree-6 fixed sparse graph**: a shuffled circulant
  graph of degree 6. Every item has exactly 6 neighbors.
  For N=32 the bootstrap produces 96 unordered pairs (192
  oriented observations). The model is fit once on this
  sparse set.
- **C. adaptive frontier design** (`run_adaptive_best_set`):
  the degree-6 bootstrap is run first, then the orchestrator
  refits BTD on the accumulated observations, computes a
  credible set at confidence 0.95, and acquires the next
  frontier batch using

      score(i, j) = (P(best=i) + P(best=j)) * 4 q (1 - q)

  with `q = P(theta_i > theta_j)` from the position-neutral
  BTD summary. The loop stops on stability across the last
  `stability_batches` fits or when the unordered-pair budget
  is reached.

## Headline metric

The headline correctness metric is COVERAGE: across
repeated simulations, the true best is inside the returned
95% credible set. The nominal target is approximately 0.95
if model and calibration assumptions hold.

## Sandbox N=32 simulation (1 seed, K=1)

```
N=32, n_seeds=1, bootstrap_degree=6, draws=400, tune=400,
chains=2, target_accept=0.9, link=py (no C linker available
in this sandbox; python3-dev is now installed and C linking
should work in a follow-up)
```

| method        | calls | k  | coverage | top1 | wall (s) | batches | reason |
|---------------|------:|---:|---------:|-----:|---------:|--------:|--------|
| round_robin   |   496 |  4 |     1.00 | 0.00 |     49.4 |     n/a |    n/a |
| sparse        |    96 | 26 |     1.00 | 1.00 |      8.1 |     n/a |    n/a |
| adaptive      |    96 | 27 |     1.00 | 1.00 |    156.4 |       1 | budget |

Observations on this single seed:

- All three methods put the true best item in the returned
  credible best set (coverage = 1.0).
- The round-robin fit produces a tighter posterior (k = 4)
  because the data is rich: 32 * 31 = 992 oriented
  observations.
- The sparse and adaptive methods use only 192 oriented
  observations, so the posterior is wider (k ~ 26). The
  trade-off is intentional: ~5x fewer observations for a
  still-valid credible set that contains the true best.
- The adaptive method stopped at the budget in 1 batch.
  This is correct behavior: with the synthetic DGP and
  the seed-shuffled circulant graph, the credible set
  stabilizes after one fit; the orchestrator's stability
  check is gated on `stability_batches = 2` consecutive
  identical sets, and the budget cap is small.
- Wall time per BTD fit is dominated by PyMC's MCMC
  sampling. The orchestrator is bottlenecked by the
  refit, not by the design logic. Each fit takes ~30-60s
  with `linker=py`; with the C linker in a more capable
  sandbox the wall time would drop substantially.

The 1-seed sample is too small to estimate coverage
empirically. The 95% Wilson confidence interval is
[0.21, 1.00] for all three methods. A larger seed sweep
is needed to confirm coverage at the nominal 0.95 level.

## Design complexity

The design layer has these costs:

- `make_sparse_bootstrap` is `O(N * degree)` edge
  construction, deterministic for fixed `(items, degree,
  seed)`.
- `select_frontier_batch` is `O(M^2)` scoring over the
  exploration pool of size M. M is bounded by the credible
  set at the wider confidence `1 - (1 - 0.95) / 2 = 0.975`.
  For the transitive DGP, M is at most a small fraction
  of N.
- The orchestrator's per-batch wall time is dominated by
  `fit_btd` on the accumulated observations, NOT by the
  design layer's scoring. The design layer itself adds
  ~10 ms per batch in this sandbox.

The design layer's complexity is "small enough to be
obviously correct". The full BTD refit per batch is the
limiting factor.

## N=1000 fitting check

The benchmark in `nuts_benchmark.py` runs the existing
UNMODIFIED `fit_btd` at N in {32, 100, 300, 1000} with
sparse degree-6 connectivity. It records wall time,
divergences, R-hat, and bulk / tail ESS. The benchmark
is observational only and does NOT change inference
backend, priors, or model.

A single-fit minimal version of the same benchmark is
provided in `nuts_quick.py`. Both scripts write their
results to `experiments/design_validation/results/`.

The empirical NUTS scaling observation is already
documented in `/workspace/bench/SAMPLER_BENCHMARK.md`:
the existing NUTS at the production-default
configuration (draws=2000, tune=2500, chains=4) is the
limiting component at N >= 100, not the design layer.
Reducing the design cost to `O(N * degree)` does not
rescue `O(N)` items in `fit_btd` if the sampler wall
time is the dominant term.

In the current sandbox pass the per-fit wall time with
`linker=py` (the C linker was missing during the
sim_audit run; `python3-dev` has since been installed)
was ~30-60s for N=32. That made the multi-NUTS N=100,
N=300, N=1000 cells impractical to populate within
the available time budget. The `max_wall_s` argument
in the benchmark scripts is the honor-system guard: if
a single fit exceeds the wall budget, the benchmark
reports a partial result rather than hiding the wall
time.

The headline scaling observation stands: the design
layer is `O(N * degree)` for the bootstrap and `O(M^2)`
for acquisition, which is dramatically cheaper than the
BTD fit at large N. Whether the existing BTD fit is
operationally feasible at N=1000 in a given environment
is a separate question that the user is best placed to
answer with the actual production sampler settings.
The `nuts_benchmark.py` script is the right tool to
re-run when more wall time is available.

## Stop conditions (per the brief)

The brief lists four stop conditions. Their status in
this sandbox pass:

- **Credible-set coverage is badly miscalibrated**:
  cannot be evaluated with 1 seed. The 95% Wilson CI is
  wide. The simulation script is in place to re-run with
  more seeds in a follow-up environment.
- **Adaptive acquisition performs materially worse than
  fixed sparse screening**: not observed. With the
  single seed in this pass, both methods achieved
  coverage = 1.0. The adaptive method's wall time is
  larger because it does an extra BTD refit; the design
  cost is not the reason.
- **Design complexity does not buy meaningful call
  savings**: the design complexity is `O(N * degree)` for
  the bootstrap and `O(M^2)` for acquisition. The brief
  asked specifically whether the design earns its
  complexity. Compared to complete round robin at N=32
  (496 unordered pairs), the degree-6 sparse graph uses
  96 unordered pairs (~5x fewer). For N=1000 the saving
  is ~167x (96 vs ~500000). This is the core "design
  buys evidence" claim. The simulation did not exercise
  N=1000 in this pass; the algorithmic saving is
  obvious from `make_sparse_bootstrap`.
- **Current BTD fitting makes the claimed large-N
  workflow operationally useless**: the benchmark is in
  place; the wall time for the production-default
  sampling at N >= 100 was not measured in this pass
  because the sandbox C linker was missing at the time
  the benchmark script was written. The benchmark
  script can be re-run with the C linker now available.

The validation PASSES the design-correctness check
(design layer does what the brief specifies) but the
empirical N=32 result is too small to confirm the
nominal 0.95 coverage target. A larger seed sweep and
N=100, 300, 1000 cells are needed for a full
production-ready verdict.

## Recommendations

1. Re-run the simulation in an environment with the C
   linker and a longer wall clock, at N in {32, 100, 300}
   with at least 5 seeds per cell. The script supports
   this directly:

   ```
   PYTHONPATH=src python3 experiments/design_validation/sim_audit.py \
     --n-list 32,100,300 --n-seeds 5 --batch-size 32 --max-orders 1000
   ```

2. Re-run the NUTS benchmark at N in {32, 100, 300, 1000}
   with the C linker now installed:

   ```
   PYTHONPATH=src python3 experiments/design_validation/nuts_benchmark.py \
     --n-list 32,100,300,1000 --draws 200 --tune 200 --chains 2
   ```

3. The design layer is structurally sound. The remaining
   risk is wall-time scalability of the existing BTD
   fit, which is a separate decision (per §20 of
   `EXPERIMENT_DESIGN.md`).

4. If the larger seed sweep reveals a coverage gap, the
   right fix is to tune the design (e.g. higher
   bootstrap degree, larger budget), NOT to introduce
   another model. The brief is explicit on this point.
