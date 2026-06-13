# Revision study (IEEE Access revision experiments)

This directory contains the code and data behind the revision experiments
added to the paper (aggregate calibration validation, percentile sweep by
exact replay, controlled same-pool baseline comparison with an oracle upper
bound, and the paired statistics). It is self-contained and additive: no
file outside this directory changed format or name (the solver gained three
optional environment switches, see below).

## Reproduction tiers

**Tier 1 — default (seconds; no dataset, no GPU).** Regenerates the
revision figures, the same-pool comparison table, and the statistics
summary from the released analysis CSVs:

```bash
python code/revision_study/run_revision_analysis.py
```

Outputs to `/results/revision_study/` (local fallback
`code/revision_study/results/revision_study/`):
`fig_calibration_aggregate.png` (paper Fig. 3),
`fig_percentile_sweep.png` (paper Fig. 4),
`table_same_pool_comparison.csv` (paper Table 3),
`revision_summary.md` (headline statistics).

**Tier 2 — re-analysis (minutes per run; FloorSet dataset, no GPU).**
Re-derives the analysis CSVs from the released recorded candidate pools by
replaying every selection policy (raw / gap(p) for 41 values of p / oracle)
through the *unmodified* official contest evaluator:

```bash
python code/revision_study/run_revision_analysis.py --reanalyze
```

The replay validates itself: the gap(0.3) replay must reproduce each run's
actual recorded selections (0 mismatches), and the evaluator costs must
match the run logs.

**Tier 3 — full re-recording (hours; FloorSet dataset + GPU).** Re-records
the candidate pools themselves with four full 100-case evaluation runs:

```bash
bash code/revision_study/run_battery.sh
```

## Contents

- `pools/run{A,B,C}_gap.jsonl.gz`, `pools/runE_raw.jsonl.gz` — recorded
  candidate pools of the four runs (one JSON line per validation instance:
  every candidate's generator tag, raw HPWL/area/V_rel, and full layout,
  plus the selected index). `pools/*.log` are the matching evaluation logs.
- `analysis/analysis_run*/` — per-candidate evaluator metrics, calibration
  ratios (Fig. 3 data), rank-alignment statistics, Prop.-1 epsilon/r
  measurements, percentile-sweep scores (Fig. 4 data), and paired
  per-case policy costs (Table 3 / statistics data), with `summary.json`.
- `analyze_pools.py` — the replay/analysis engine (Tier 2).
- `run_revision_analysis.py` — Tier 1/2 entry point.
- `run_battery.sh` — Tier 3 entry point.
- `make_paper_artifacts.py` — provenance copy of the script that computed
  the manuscript's numbers/figures/tables from these CSVs (its `--fill`
  mode targets the paper LaTeX tree and is not runnable in the capsule).

## Solver environment switches (added for this study)

`code/project_code/gap_aware_reranking_floorplanning/src/solver.py` now
supports three optional environment variables (default behaviour is
unchanged when they are unset):

- `RERANK_DUMP=<path>` — append one JSON record per solved instance with
  the complete feasible-candidate pool (used to record the pools above);
- `RERANK_MODE=raw` — final candidate selection by the raw-magnitude cost
  (paper Eq. 4) instead of the gap-aware cost (the revision's controlled
  baseline, run E);
- `RERANK_PCT=<p>` — rerank baseline percentile (default 0.3).

## Headline numbers reproduced by this study

On identical recorded pools, scored by the official evaluator:
raw rerank S100 = 1.42551, gap-aware S100 = 1.40228, oracle = 1.40228
(the gap-aware rule recovers 99.99% of the available selection headroom);
paired Wilcoxon p = 3.7e-9 (38 wins / 60 ties / 2 losses); per-instance
Kendall tau median 0.78 (raw) vs 0.94 (gap); S100(p) constant to within
3e-6 for all p in [0,1] (the percentile is not a tunable hyperparameter).
