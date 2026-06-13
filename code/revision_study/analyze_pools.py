#!/usr/bin/env python3
"""
Offline rerank-policy replay and calibration analysis over recorded
candidate pools (revision study for the IEEE Access paper).

Inputs : a RERANK_DUMP jsonl (or .jsonl.gz) file -- one record per case with
         the full candidate pool including layouts -- plus the matching
         quick_eval verbose log (per-case runtimes).
Outputs: per-candidate evaluator metrics, policy-replay scores (raw / gap(p) /
         oracle), percentile sweep curves (S100 and S15), per-family
         calibration ratios, per-case rank correlations, Prop-1 epsilon stats,
         and paired statistics (Wilcoxon + bootstrap CI).

Every candidate is scored by the *official contest evaluator* (unmodified),
so replayed selection policies are exactly what an end-to-end run with that
policy would produce (selection happens after all generators have run, and
per-case runtime does not depend on which candidate is selected).

Requires the public FloorSet repository (set FLOORSET_ROOT, or place it at
/data/FloorSet or next to the capsule).
"""
import argparse
import gzip
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).parent.resolve()

# FloorSet location: env var, then various fallbacks (mirrors quick_eval.py).
_env = os.environ.get("FLOORSET_ROOT")
_candidates = ([Path(_env)] if _env else []) + [
    Path("/data/FloorSet"),
    *[p / "FloorSet" for p in THIS_DIR.parents],
    Path.cwd() / "FloorSet",
]
FLOORSET_ROOT = next(
    (p for p in _candidates if (p / "iccad2026contest").exists()),
    _candidates[0])
CONTEST_DIR = FLOORSET_ROOT / "iccad2026contest"
sys.path.insert(0, str(FLOORSET_ROOT))
sys.path.insert(0, str(CONTEST_DIR))

from iccad2026_evaluate import (  # noqa: E402
    evaluate_solution, compute_cost, compute_total_score, M_PENALTY,
)
from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402

SHORT_IDS = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99]

# ---------------------------------------------------------------- dataset


def extract_baseline(inputs, labels, block_count):
    """Identical to quick_eval.extract_baseline (kept in sync)."""
    import torch  # noqa: F401
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    polygons, metrics = labels
    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append((float(x_min), float(y_min),
                              float(x_max - x_min), float(y_max - y_min)))
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    area = None
    hpwl_b2b_base = None
    hpwl_p2b_base = None
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0:
            hpwl_b2b_base = float(metrics[-2])
        if metrics[-1] >= 0:
            hpwl_p2b_base = float(metrics[-1])
    if area is None:
        x_min = min(p[0] for p in positions)
        y_min = min(p[1] for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)
        area = (x_max - x_min) * (y_max - y_min)
    if hpwl_b2b_base is None or hpwl_p2b_base is None:
        from iccad2026_evaluate import calculate_hpwl_b2b, calculate_hpwl_p2b
        hpwl_b2b_base = calculate_hpwl_b2b(positions, b2b_conn)
        hpwl_p2b_base = calculate_hpwl_p2b(positions, p2b_conn, pins_pos)
    return {
        "hpwl_baseline": hpwl_b2b_base + hpwl_p2b_base,
        "area_baseline": area,
    }, positions

# ---------------------------------------------------------------- helpers


def read_dump_lines(path: Path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return f.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def solver_pct(xs, p):
    """Replicates solver.py's _pct exactly (index = int(len*p), clamped)."""
    if not xs:
        return 1.0
    i = max(0, min(len(xs) - 1, int(len(xs) * p)))
    return xs[i]


def family_of(tag):
    if tag == "selected":
        return "selected"
    return "DiT" if tag.startswith("dit-") else "classical"


def gap_costs(cands, p):
    """Solver-side gap-aware rerank costs at percentile p (exact replica)."""
    hs = sorted(c["h"] for c in cands if c["h"] > 0)
    as_ = sorted(c["a"] for c in cands if c["a"] > 0)
    base_h = solver_pct(hs, p)
    base_a = solver_pct(as_, p)
    out = []
    for c in cands:
        h_gap = (c["h"] / max(base_h, 1e-6)) - 1.0
        a_gap = (c["a"] / max(base_a, 1e-6)) - 1.0
        out.append((1.0 + 0.5 * (h_gap + a_gap)) * math.exp(2.0 * c["v"]))
    return out, base_h, base_a


def raw_costs(cands):
    """Solver-side raw-magnitude rerank costs (paper Eq. 4)."""
    return [(c["h"] + 0.5 * c["a"]) * math.exp(2.0 * c["v"]) for c in cands]


def kendall_tau(x, y):
    from scipy.stats import kendalltau
    t, _ = kendalltau(x, y)
    return float(t) if t == t else float("nan")

# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", help="RERANK_DUMP jsonl or jsonl.gz path")
    ap.add_argument("log", help="matching quick_eval --verbose log")
    ap.add_argument("--out", default=None, help="output dir")
    args = ap.parse_args()

    dump_path = Path(args.dump)
    stem = dump_path.name.replace(".jsonl.gz", "").replace(".jsonl", "")
    out_dir = Path(args.out) if args.out else THIS_DIR / ("analysis_" + stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-case runtimes + reported metrics from verbose log
    case_rt = {}
    case_logged = {}
    pat = re.compile(
        r"case\s+(\d+)\s+n=\s*(\d+)\s+(\S+)\s+hpwl_gap=([+-][\d.]+)\s+"
        r"area_gap=([+-][\d.]+)\s+v_rel=([\d.]+)\s+rt=\s*([\d.]+)s\s+cost=([\d.]+)")
    for line in Path(args.log).read_text().splitlines():
        m = pat.search(line)
        if m:
            cid = int(m.group(1))
            case_rt[cid] = float(m.group(7))
            case_logged[cid] = dict(
                n=int(m.group(2)), flag=m.group(3),
                hpwl_gap=float(m.group(4)), area_gap=float(m.group(5)),
                v_rel=float(m.group(6)), cost=float(m.group(8)))
    print(f"parsed {len(case_rt)} case runtimes from log")
    ids = sorted(case_rt.keys())
    median_rt = sorted(case_rt.values())[len(case_rt) // 2]

    # ---- dataset
    print(f"loading dataset from {FLOORSET_ROOT} ...")
    dataset = FloorplanDatasetLiteTest(str(FLOORSET_ROOT) + "/")

    # ---- dump records (in case order of the run)
    recs = [json.loads(l) for l in read_dump_lines(dump_path)]
    assert len(recs) == len(ids), f"dump has {len(recs)} records vs {len(ids)} log cases"

    per_cand = []          # rows for per-candidate csv
    per_case = {}          # cid -> dict with pool arrays
    eps_rows = []

    for k, cid in enumerate(ids):
        rec = recs[k]
        sample = dataset[cid]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        assert block_count == rec["block_count"], f"case {cid}: n mismatch"
        baseline, target_pos = extract_baseline(inputs, labels, block_count)
        rt = case_rt[cid]
        rfac = rt / max(median_rt, 0.01)

        cands = rec["candidates"]
        eval_rows = []
        for ci, c in enumerate(cands):
            m = evaluate_solution(
                {"positions": [tuple(p) for p in c["pos"]], "runtime": rt},
                baseline, constraints, b2b_conn, p2b_conn, pins_pos,
                area_target, target_pos, median_runtime=1.0)
            cost = compute_cost(m.hpwl_gap, m.area_gap, m.violations_relative,
                                rfac, m.is_feasible)
            row = dict(
                case=cid, n=block_count, idx=ci, tag=c["tag"],
                family=family_of(c["tag"]),
                h_solver=c["h"], a_solver=c["a"], v_solver=c["v"],
                hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                v_rel=m.violations_relative, feasible=bool(m.is_feasible),
                cost_contest=cost,
                hpwl_eval=m.hpwl_gap * baseline["hpwl_baseline"] + baseline["hpwl_baseline"],
                area_eval=m.area_gap * baseline["area_baseline"] + baseline["area_baseline"],
            )
            eval_rows.append(row)
            per_cand.append(row)

        # epsilon for Prop 1 (evaluator-side HPWL/area, percentile p=0.3)
        hs_eval = sorted(r["hpwl_eval"] for r in eval_rows if r["tag"] != "selected")
        as_eval = sorted(r["area_eval"] for r in eval_rows if r["tag"] != "selected")
        H0 = solver_pct(hs_eval, 0.3)
        A0 = solver_pct(as_eval, 0.3)
        eps_rows.append(dict(
            case=cid, n=block_count,
            eps_h=H0 / baseline["hpwl_baseline"] - 1.0,
            eps_a=A0 / baseline["area_baseline"] - 1.0,
            hpwl_star=baseline["hpwl_baseline"],
            area_star=baseline["area_baseline"],
            max_abs_dh=max(abs(r["hpwl_gap"]) for r in eval_rows),
            max_abs_da=max(abs(r["area_gap"]) for r in eval_rows),
        ))

        per_case[cid] = dict(rec=rec, eval_rows=eval_rows, rt=rt,
                             n=block_count)
        if (k + 1) % 20 == 0:
            print(f"  evaluated {k + 1}/{len(ids)} cases")

    # ---- policy replay
    policies = {}

    def replay(select_fn, name):
        costs, blocks, picks, feas = [], [], {}, 0
        for cid in ids:
            pc = per_case[cid]
            sel = select_fn(pc)
            row = pc["eval_rows"][sel]
            costs.append(row["cost_contest"])
            blocks.append(pc["n"])
            picks[cid] = sel
            feas += int(row["feasible"])
        s100 = compute_total_score(costs, blocks)
        # S15 on the short subset with subset-median runtime factors
        if not all(c in case_rt for c in SHORT_IDS):
            policies[name] = dict(name=name, s100=s100, s15=float("nan"),
                                  feasible=feas, picks=picks,
                                  costs=dict(zip(ids, costs)))
            return policies[name]
        sub_rt = sorted(case_rt[c] for c in SHORT_IDS)
        sub_med = sub_rt[len(sub_rt) // 2]
        sub_costs, sub_blocks = [], []
        for cid in SHORT_IDS:
            pc = per_case[cid]
            row = pc["eval_rows"][picks[cid]]
            c15 = compute_cost(row["hpwl_gap"], row["area_gap"], row["v_rel"],
                               case_rt[cid] / max(sub_med, 0.01),
                               row["feasible"])
            sub_costs.append(c15)
            sub_blocks.append(pc["n"])
        s15 = compute_total_score(sub_costs, sub_blocks)
        policies[name] = dict(name=name, s100=s100, s15=s15,
                              feasible=feas, picks=picks,
                              costs=dict(zip(ids, costs)))
        return policies[name]

    def sel_raw(pc):
        rc = raw_costs(pc["rec"]["candidates"])
        return int(np.argmin(rc))

    def sel_gap(p):
        def f(pc):
            gc, _, _ = gap_costs(pc["rec"]["candidates"], p)
            return int(np.argmin(gc))
        return f

    def sel_oracle(pc):
        return int(np.argmin([r["cost_contest"] for r in pc["eval_rows"]]))

    replay(sel_raw, "raw")
    replay(sel_oracle, "oracle")
    sweep_ps = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.025)]
    for p in sweep_ps:
        replay(sel_gap(p), f"gap_{p:.3f}")

    # validate replay against the actual run's recorded selection
    mode = recs[0].get("mode", "gap")
    ref_policy = "raw" if mode == "raw" else f"gap_{recs[0]['pct']:.3f}"
    mism = [cid for k, cid in enumerate(ids)
            if policies[ref_policy]["picks"][cid] != recs[k]["selected_idx"]]
    print(f"replay-vs-actual selection mismatches ({ref_policy}): "
          f"{len(mism)}/{len(ids)} {mism[:10]}")

    # validate evaluator metrics of actual selection vs logged metrics.
    # NB: the verbose log prints cost with median_runtime=1.0 (pre-recompute),
    # so compare at that runtime normalisation.
    max_dc = 0.0
    for k, cid in enumerate(ids):
        row = per_case[cid]["eval_rows"][recs[k]["selected_idx"]]
        c_log_norm = compute_cost(row["hpwl_gap"], row["area_gap"],
                                  row["v_rel"], case_rt[cid], row["feasible"])
        max_dc = max(max_dc, abs(c_log_norm - case_logged[cid]["cost"]))
    print(f"max |evaluator cost - logged cost| on actual selections: {max_dc:.6f}")

    # ---- calibration ratios + rank correlations
    cal_rows = []
    tau_rows = []
    for cid in ids:
        pc = per_case[cid]
        rows = [r for r in pc["eval_rows"] if r["tag"] != "selected"]
        cands = [c for c in pc["rec"]["candidates"] if c["tag"] != "selected"]
        feas = [i for i, r in enumerate(rows) if r["feasible"]]
        if len(feas) < 3:
            continue
        rc = raw_costs(cands)
        gc, _, _ = gap_costs(cands, 0.3)
        cc = [r["cost_contest"] for r in rows]
        mean_rc = np.mean([rc[i] for i in feas])
        mean_gc = np.mean([gc[i] for i in feas])
        mean_cc = np.mean([cc[i] for i in feas])
        for i in feas:
            cal_rows.append(dict(
                case=cid, n=pc["n"], tag=rows[i]["tag"],
                family=rows[i]["family"],
                v_rel=rows[i]["v_rel"],
                rho_raw=(rc[i] / mean_rc) / (cc[i] / mean_cc),
                rho_gap=(gc[i] / mean_gc) / (cc[i] / mean_cc),
            ))
        tau_rows.append(dict(
            case=cid, n=pc["n"],
            tau_raw=kendall_tau([rc[i] for i in feas], [cc[i] for i in feas]),
            tau_gap=kendall_tau([gc[i] for i in feas], [cc[i] for i in feas]),
            max_dh=max(rows[i]["hpwl_gap"] for i in feas),
            top1_raw=int(np.argmin([rc[i] for i in feas]) == np.argmin([cc[i] for i in feas])),
            top1_gap=int(np.argmin([gc[i] for i in feas]) == np.argmin([cc[i] for i in feas])),
        ))

    # ---- paired statistics raw vs gap(0.3) on identical pools
    from scipy.stats import wilcoxon
    raw_c = np.array([policies["raw"]["costs"][c] for c in ids])
    gap_c = np.array([policies["gap_0.300"]["costs"][c] for c in ids])
    blocks = np.array([per_case[c]["n"] for c in ids])
    diffs = raw_c - gap_c
    nz = diffs[diffs != 0]
    w_stat, w_p = (wilcoxon(nz) if len(nz) >= 5 else (float("nan"),) * 2)

    rng = np.random.default_rng(20260610)
    boot = []
    idx = np.arange(len(ids))
    for _ in range(10000):
        bs = rng.choice(idx, size=len(idx), replace=True)
        s_raw = compute_total_score(list(raw_c[bs]), list(blocks[bs]))
        s_gap = compute_total_score(list(gap_c[bs]), list(blocks[bs]))
        boot.append(s_raw - s_gap)
    boot = np.array(boot)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    # ---- write outputs
    import pandas as pd
    pd.DataFrame(per_cand).to_csv(out_dir / "per_candidate.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_ratios.csv", index=False)
    pd.DataFrame(tau_rows).to_csv(out_dir / "rank_alignment.csv", index=False)
    pd.DataFrame(eps_rows).to_csv(out_dir / "prop1_epsilon.csv", index=False)
    pol_rows = [dict(name=p["name"], s100=p["s100"], s15=p["s15"],
                     feasible=p["feasible"]) for p in policies.values()]
    pd.DataFrame(pol_rows).to_csv(out_dir / "policy_scores.csv", index=False)
    percase_rows = [dict(case=c, n=int(blocks[i]),
                         cost_raw=float(raw_c[i]), cost_gap=float(gap_c[i]),
                         cost_oracle=float(policies["oracle"]["costs"][c]),
                         rt=case_rt[c])
                    for i, c in enumerate(ids)]
    pd.DataFrame(percase_rows).to_csv(out_dir / "per_case_policy_costs.csv", index=False)

    summary = dict(
        dump=str(dump_path), mode=mode, n_cases=len(ids),
        median_rt=median_rt,
        s100_actual_from_log=None,
        replay_mismatches=len(mism),
        max_cost_check_delta=max_dc,
        s100=dict(raw=policies["raw"]["s100"],
                  gap_030=policies["gap_0.300"]["s100"],
                  oracle=policies["oracle"]["s100"]),
        s15=dict(raw=policies["raw"]["s15"],
                 gap_030=policies["gap_0.300"]["s15"],
                 oracle=policies["oracle"]["s15"]),
        wilcoxon=dict(stat=float(w_stat), p=float(w_p), n_nonzero=int(len(nz))),
        bootstrap_delta_s100=dict(mean=float(boot.mean()), ci95=ci),
        calibration=dict(
            rho_raw_dit=_qstats([r["rho_raw"] for r in cal_rows if r["family"] == "DiT"]),
            rho_raw_classical=_qstats([r["rho_raw"] for r in cal_rows if r["family"] == "classical"]),
            rho_gap_dit=_qstats([r["rho_gap"] for r in cal_rows if r["family"] == "DiT"]),
            rho_gap_classical=_qstats([r["rho_gap"] for r in cal_rows if r["family"] == "classical"]),
        ),
        tau=dict(
            raw=_qstats([t["tau_raw"] for t in tau_rows]),
            gap=_qstats([t["tau_gap"] for t in tau_rows]),
            top1_raw=float(np.mean([t["top1_raw"] for t in tau_rows])),
            top1_gap=float(np.mean([t["top1_gap"] for t in tau_rows])),
        ),
        epsilon=dict(
            eps_h=_qstats([e["eps_h"] for e in eps_rows]),
            eps_a=_qstats([e["eps_a"] for e in eps_rows]),
        ),
        sweep=[dict(p=p, s100=policies[f"gap_{p:.3f}"]["s100"],
                    s15=policies[f"gap_{p:.3f}"]["s15"]) for p in sweep_ps],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sweep"}, indent=2))
    print("sweep (p, S100, S15):")
    for s in summary["sweep"]:
        print(f"  p={s['p']:.3f}  S100={s['s100']:.5f}  S15={s['s15']:.5f}")
    print(f"outputs in {out_dir}")


def _qstats(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return None
    a = np.array(xs)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), q25=float(np.percentile(a, 25)),
                median=float(np.median(a)), q75=float(np.percentile(a, 75)),
                max=float(a.max()))


if __name__ == "__main__":
    main()
