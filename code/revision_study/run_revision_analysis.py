#!/usr/bin/env python3
"""
Revision-study entry point (IEEE Access revision experiments).

Default run (no dataset, no GPU, finishes in seconds):
  reads the released analysis CSVs under analysis/ and regenerates the
  revision paper's Figure 3 (aggregate calibration box plots), Figure 4
  (percentile sweep), Table 3 (controlled same-pool comparison), and a
  statistics summary, into <results>/revision_study/.

Optional deeper tiers:
  --reanalyze   re-derive the analysis CSVs from the released recorded
                candidate pools (pools/*.jsonl.gz) by replaying selection
                policies through the official contest evaluator. Requires
                the public FloorSet repository (FLOORSET_ROOT) but no GPU.
                Takes a few minutes per run.
  (full re-recording of the pools themselves: see run_battery.sh; GPU,
   hours.)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = ["runA_gap", "runB_gap", "runC_gap", "runE_raw"]


def _resolve_results_root() -> Path:
    for env_name in ("RESULTS_DIR", "CODEOCEAN_RESULTS_DIR"):
        raw = os.environ.get(env_name)
        if raw:
            return Path(raw)
    codeocean_results = Path("/results")
    if os.name != "nt" and codeocean_results.exists():
        return codeocean_results
    return HERE / "results"


OUTPUT_ROOT = _resolve_results_root() / "revision_study"


def analysis_dir(run: str) -> Path:
    return HERE / "analysis" / f"analysis_{run}"


def reanalyze() -> None:
    for run in RUNS:
        dump = HERE / "pools" / f"{run}.jsonl.gz"
        log = HERE / "pools" / f"{run}.log"
        out = analysis_dir(run)
        print(f"[reanalyze] {run} ...", flush=True)
        subprocess.run(
            [sys.executable, str(HERE / "analyze_pools.py"),
             str(dump), str(log), "--out", str(out)],
            check=True)


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_summary(run: str) -> dict:
    return json.loads((analysis_dir(run) / "summary.json").read_text())


def plot_calibration_boxes() -> Path:
    rows = load_csv(analysis_dir("runA_gap") / "calibration_ratios.csv")
    groups = {
        "raw,\nDiT": [float(r["rho_raw"]) for r in rows if r["family"] == "DiT"],
        "raw,\nclassical": [float(r["rho_raw"]) for r in rows if r["family"] == "classical"],
        "gap,\nDiT": [float(r["rho_gap"]) for r in rows if r["family"] == "DiT"],
        "gap,\nclassical": [float(r["rho_gap"]) for r in rows if r["family"] == "classical"],
    }
    plt.figure(figsize=(7, 4.5))
    bp = plt.boxplot(groups.values(), labels=groups.keys(), whis=(5, 95),
                     showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["#F58518", "#4C78A8", "#F58518", "#4C78A8"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    plt.axhline(1.0, linestyle="--", color="black", linewidth=1,
                label="calibration target")
    plt.axvline(2.5, color="grey", linewidth=0.8)
    plt.ylabel(r"normalised rank ratio $\rho$")
    plt.title("Aggregate calibration: raw cost (left) vs gap-aware cost (right)\n"
              "all 100 validation instances, both candidate families")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = OUTPUT_ROOT / "fig_calibration_aggregate.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def plot_percentile_sweep() -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    colors = {"runA_gap": "#4C78A8", "runB_gap": "#E45756", "runC_gap": "#54A24B"}
    raw_levels = load_summary("runA_gap")
    for run in ("runA_gap", "runB_gap", "runC_gap"):
        sweep = load_summary(run)["sweep"]
        ps = [s["p"] for s in sweep]
        axes[0].plot(ps, [s["s15"] for s in sweep], color=colors[run], label=f"{run} replay")
        axes[1].plot(ps, [s["s100"] for s in sweep], color=colors[run])
    axes[0].axhline(raw_levels["s15"]["raw"], linestyle=":", color="black",
                    label="raw policy (run A)")
    axes[1].axhline(raw_levels["s100"]["raw"], linestyle=":", color="black")
    for ax, ylab in zip(axes, ("S15", "S100")):
        ax.axvline(0.3, linestyle="--", color="grey", linewidth=1)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("rerank percentile p")
    axes[0].set_title("Percentile sweep by exact replay on frozen candidate pools\n"
                      "(flat: selection is invariant to p; dashed = deployed p=0.3)")
    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    out = OUTPUT_ROOT / "fig_percentile_sweep.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def write_same_pool_table() -> Path:
    out = OUTPUT_ROOT / "table_same_pool_comparison.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pool", "raw_S100", "gap_aware_S100", "oracle_S100",
                    "headroom_recovered_pct"])
        for run in RUNS:
            s = load_summary(run)["s100"]
            rec = (s["raw"] - s["gap_030"]) / (s["raw"] - s["oracle"]) * 100
            w.writerow([run, f"{s['raw']:.5f}", f"{s['gap_030']:.5f}",
                        f"{s['oracle']:.5f}", f"{rec:.2f}"])
    return out


def write_stats_summary() -> Path:
    sA = load_summary("runA_gap")
    rows = load_csv(analysis_dir("runA_gap") / "per_case_policy_costs.csv")
    nw = sum(float(r["cost_raw"]) > float(r["cost_gap"]) + 1e-9 for r in rows)
    nl = sum(float(r["cost_gap"]) > float(r["cost_raw"]) + 1e-9 for r in rows)
    sweep_var = max(
        max(s["s100"] for s in load_summary(r)["sweep"])
        - min(s["s100"] for s in load_summary(r)["sweep"])
        for r in ("runA_gap", "runB_gap", "runC_gap"))
    lines = [
        "# Revision-study summary",
        "",
        "Same-pool comparison (run A pools, official evaluator):",
        f"- raw rerank        S100 = {sA['s100']['raw']:.5f}",
        f"- gap-aware rerank  S100 = {sA['s100']['gap_030']:.5f}",
        f"- oracle            S100 = {sA['s100']['oracle']:.5f}",
        "",
        f"Paired per-instance: gap better on {nw}, worse on {nl}, "
        f"tie on {len(rows) - nw - nl} of {len(rows)}.",
        f"Wilcoxon signed-rank: W = {sA['wilcoxon']['stat']:.0f}, "
        f"p = {sA['wilcoxon']['p']:.2e} (n_nonzero = {sA['wilcoxon']['n_nonzero']}).",
        f"Bootstrap dS100 = {sA['bootstrap_delta_s100']['mean']:.4f}, "
        f"95% CI [{sA['bootstrap_delta_s100']['ci95'][0]:.4f}, "
        f"{sA['bootstrap_delta_s100']['ci95'][1]:.4f}].",
        "",
        "Rank alignment (per-instance, run A):",
        f"- Kendall tau median: raw {sA['tau']['raw']['median']:.2f} -> "
        f"gap {sA['tau']['gap']['median']:.2f}",
        f"- top-1 agreement with oracle: raw {sA['tau']['top1_raw']*100:.0f}% -> "
        f"gap {sA['tau']['top1_gap']*100:.0f}%",
        "",
        f"Percentile sweep: S100(p) varies by at most {sweep_var:.2e} over the",
        "entire range p in [0,1] on all three gap-aware pools (selection is",
        "invariant to the percentile; see fig_percentile_sweep.png).",
        "",
        "Replay validation: the gap(0.3) replay reproduces each run's actual",
        f"selections on all instances ({sA['replay_mismatches']} mismatches on run A).",
    ]
    out = OUTPUT_ROOT / "revision_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reanalyze", action="store_true",
                    help="re-derive analysis CSVs from the recorded pools "
                         "(requires the FloorSet dataset)")
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.reanalyze:
        reanalyze()

    missing = [r for r in RUNS if not (analysis_dir(r) / "summary.json").exists()]
    if missing:
        raise FileNotFoundError(f"analysis outputs missing for: {missing} "
                                "(run with --reanalyze or restore analysis/)")

    outputs = [
        plot_calibration_boxes(),
        plot_percentile_sweep(),
        write_same_pool_table(),
        write_stats_summary(),
    ]
    print("Revision-study outputs:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
