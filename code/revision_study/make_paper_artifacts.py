#!/usr/bin/env python3
# PROVENANCE COPY. This script computed every number, figure, and table in
# Sections V-D, V-E, and VII of the revised manuscript from the analysis
# CSVs in analysis/. Its --fill mode targets the paper's LaTeX source tree
# and is not runnable inside the capsule; it is included so reviewers can
# audit exactly how the manuscript numbers were derived. For a capsule-
# runnable regeneration of the same figures and statistics, use
# run_revision_analysis.py instead.
"""
Compute all @@PLACEHOLDER@@ values, figure pgfplots code, and the same-pool
table body for the revised paper from the analyze_pools.py outputs.

Usage:
  python make_paper_artifacts.py            # compute + write placeholder JSONs
  python make_paper_artifacts.py --fill     # additionally fill paper.tex
                                            # (from paper_template.tex snapshot)

Runs expected under revision_runs/analysis_runX_gap / runE_raw (A required).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PAPER_DIR = (HERE.parent / "submission" / "project_code" /
             "gap_aware_reranking_floorplanning" / "paper")


def load_run(name):
    d = HERE / "analysis" / f"analysis_{name}"
    if not (d / "summary.json").exists():
        return None
    out = {"summary": json.loads((d / "summary.json").read_text())}
    for f in ("per_candidate", "calibration_ratios", "rank_alignment",
              "prop1_epsilon", "per_case_policy_costs"):
        p = d / f"{f}.csv"
        if p.exists():
            out[f] = pd.read_csv(p)
    return out


def fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true")
    args = ap.parse_args()

    runs = {}
    for name in ("runA_gap", "runB_gap", "runC_gap", "runE_raw"):
        r = load_run(name)
        if r:
            runs[name] = r
    assert "runA_gap" in runs, "run A analysis missing"
    A = runs["runA_gap"]
    gap_runs = [runs[k] for k in ("runA_gap", "runB_gap", "runC_gap")
                if k in runs]
    print("loaded runs:", list(runs.keys()))

    L = {}   # latex placeholders
    P = {}   # plain-text placeholders (for the docx)

    # ---------- candidate counts & A1
    pc = A["per_candidate"]
    pc_nosel = pc[pc.tag != "selected"]
    ncand = len(pc_nosel)
    L["@@NCAND@@"] = P["@@NCAND@@"] = f"{ncand}"
    all_pc = pd.concat([r["per_candidate"] for r in runs.values()])
    all_pc = all_pc[all_pc.tag != "selected"]
    L["@@NCANDALL@@"] = P["@@NCANDALL@@"] = f"{len(all_pc)}"
    neg = ((all_pc.hpwl_gap < 0) | (all_pc.area_gap < 0)).sum()
    a1pct = 100.0 * (1 - neg / len(all_pc))
    L["@@A1PCT@@"] = (f"$100\\%$" if neg == 0 else f"${a1pct:.2f}\\%$")
    P["@@A1PCT@@"] = ("100%" if neg == 0 else f"{a1pct:.2f}%")
    print(f"A1: {neg} negative-gap candidates of {len(all_pc)}")

    # ---------- kappa / r stats (run A, p=0.3)
    eps = A["prop1_epsilon"].copy()
    eps["kh"] = 1 + eps.eps_h
    eps["ka"] = 1 + eps.eps_a
    eps["r"] = eps.kh / eps.ka
    eps["r_raw"] = eps.area_star / (2 * eps.hpwl_star)

    def med_iqr(s, nd=2):
        return (fmt(s.median(), nd),
                f"[{fmt(s.quantile(.25), nd)}, {fmt(s.quantile(.75), nd)}]")

    khm, khi = med_iqr(eps.kh)
    kam, kai = med_iqr(eps.ka)
    rm, ri = med_iqr(eps.r)
    L["@@KHMED@@"] = f"${khm}$";  P["@@KHMED@@"] = khm
    L["@@KHIQR@@"] = f"${khi}$";  P["@@KHIQR@@"] = khi
    L["@@KAMED@@"] = f"${kam}$";  P["@@KAMED@@"] = kam
    L["@@KAIQR@@"] = f"${kai}$";  P["@@KAIQR@@"] = kai
    L["@@RMED@@"] = f"${rm}$";    P["@@RMED@@"] = rm
    L["@@RIQR@@"] = f"${ri}$";    P["@@RIQR@@"] = ri
    # bound covering 90% of instances (template embeds this inside math mode)
    rb = np.quantile(np.abs(eps.r - 1), 0.9)
    L["@@RBOUND@@"] = fmt(rb, 2); P["@@RBOUND@@"] = fmt(rb, 2)
    L["@@RPCT@@"] = "$90\\%$";    P["@@RPCT@@"] = "90%"
    rrm = eps.r_raw.median()
    L["@@RRAWMED@@"] = f"{fmt(rrm, 1)}"; P["@@RRAWMED@@"] = fmt(rrm, 1)
    rrr = f"[{fmt(eps.r_raw.min(), 1)}, {fmt(eps.r_raw.max(), 1)}]"
    L["@@RRAWRANGE@@"] = f"${rrr}$"; P["@@RRAWRANGE@@"] = rrr

    # ---------- calibration rho stats (run A)
    cal = A["calibration_ratios"]
    g = lambda fam, col: cal[cal.family == fam][col]
    L["@@RHORAWDIT@@"] = f"${fmt(g('DiT','rho_raw').median())}$"
    L["@@RHORAWCLS@@"] = f"${fmt(g('classical','rho_raw').median())}$"
    L["@@RHOGAPDIT@@"] = f"${fmt(g('DiT','rho_gap').median())}$"
    L["@@RHOGAPCLS@@"] = f"${fmt(g('classical','rho_gap').median())}$"
    for k in ("@@RHORAWDIT@@", "@@RHORAWCLS@@", "@@RHOGAPDIT@@", "@@RHOGAPCLS@@"):
        P[k] = L[k].strip("$")
    iqr_raw = (cal.groupby("family").rho_raw
               .apply(lambda s: s.quantile(.75) - s.quantile(.25)).mean())
    iqr_gap = (cal.groupby("family").rho_gap
               .apply(lambda s: s.quantile(.75) - s.quantile(.25)).mean())
    shrink = (1 - iqr_gap / iqr_raw) * 100
    L["@@RHOSHRINK@@"] = f"${shrink:.0f}\\%$ (mean per-family IQR)"
    P["@@RHOSHRINK@@"] = f"{shrink:.0f}% (mean per-family IQR)"

    # ---------- tau / top1 (run A)
    tau = A["rank_alignment"]
    trm, tri = med_iqr(tau.tau_raw)
    tgm, tgi = med_iqr(tau.tau_gap)
    L["@@TAURAWMED@@"] = f"${trm}$"; P["@@TAURAWMED@@"] = trm
    L["@@TAURAWIQR@@"] = f"${tri}$"; P["@@TAURAWIQR@@"] = tri
    L["@@TAUGAPMED@@"] = f"${tgm}$"; P["@@TAUGAPMED@@"] = tgm
    L["@@TAUGAPIQR@@"] = f"${tgi}$"; P["@@TAUGAPIQR@@"] = tgi
    t1r = tau.top1_raw.mean(); t1g = tau.top1_gap.mean()
    L["@@TOP1RAW@@"] = f"${t1r*100:.0f}\\%$"; P["@@TOP1RAW@@"] = f"{t1r*100:.0f}%"
    L["@@TOP1GAP@@"] = f"${t1g*100:.0f}\\%$"; P["@@TOP1GAP@@"] = f"{t1g*100:.0f}%"

    # case 70
    if (eps.case == 70).any():
        e70 = eps[eps.case == 70].iloc[0]
        t70 = tau[tau.case == 70].iloc[0]
        L["@@R70@@"] = fmt(e70.r); P["@@R70@@"] = fmt(e70.r)
        L["@@TAU70@@"] = fmt(t70.tau_gap); P["@@TAU70@@"] = fmt(t70.tau_gap)
        L["@@TAURAW70@@"] = fmt(t70.tau_raw); P["@@TAURAW70@@"] = fmt(t70.tau_raw)
    else:
        print("WARNING: case 70 missing (smoke test?)")
        for k in ("@@R70@@", "@@TAU70@@", "@@TAURAW70@@"):
            L[k] = P[k] = "0.00"

    # ---------- policies / same-pool stats
    def s100(run, pol):
        return run["summary"]["s100"][pol]

    raws = [s100(r, "raw") for r in gap_runs]
    gaps = [s100(r, "gap_030") for r in gap_runs]
    oras = [s100(r, "oracle") for r in gap_runs]
    L["@@RAWS100@@"] = P["@@RAWS100@@"] = fmt(raws[0], 5)
    L["@@GAPS100@@"] = P["@@GAPS100@@"] = fmt(gaps[0], 5)
    L["@@ORACLES100@@"] = P["@@ORACLES100@@"] = fmt(oras[0], 5)
    headroom = raws[0] - oras[0]
    recover = (raws[0] - gaps[0]) / headroom * 100
    L["@@HEADROOM@@"] = fmt(headroom, 3); P["@@HEADROOM@@"] = fmt(headroom, 3)
    L["@@RECOVERPCT@@"] = f"${recover:.2f}\\%$"; P["@@RECOVERPCT@@"] = f"{recover:.2f}%"

    w = A["summary"]["wilcoxon"]
    pstr = f"{w['p']:.1e}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-")
    if "10^{-" in pstr:
        pstr += "}"
    L["@@WILCOXSTATS@@"] = (f"$W = {w['stat']:.0f}$, $p = {pstr}$, "
                            f"$n_{{\\neq}} = {w['n_nonzero']}$")
    P["@@WILCOXSTATS@@"] = (f"W = {w['stat']:.0f}, p = {w['p']:.1e}, with "
                            f"{w['n_nonzero']} non-tied pairs")
    ppc = A["per_case_policy_costs"]
    nw = int((ppc.cost_raw > ppc.cost_gap + 1e-9).sum())
    nl = int((ppc.cost_gap > ppc.cost_raw + 1e-9).sum())
    nt = 100 - nw - nl
    L["@@NWINS@@"] = P["@@NWINS@@"] = str(nw)
    L["@@NLOSS@@"] = P["@@NLOSS@@"] = str(nl)
    L["@@NTIES@@"] = P["@@NTIES@@"] = str(nt)
    boot = A["summary"]["bootstrap_delta_s100"]
    L["@@BOOTMEAN@@"] = P["@@BOOTMEAN@@"] = fmt(boot["mean"], 4)
    ci = boot["ci95"]
    L["@@BOOTCI@@"] = P["@@BOOTCI@@"] = f"[{fmt(ci[0],4)}, {fmt(ci[1],4)}]"


    # ---------- sweep stats
    sweeps = [pd.DataFrame(r["summary"]["sweep"]) for r in gap_runs]
    # total variation of each score across the full p range, worst over runs
    sweep_var = max(max(sw.s100.max() - sw.s100.min(),
                        sw.s15.max() - sw.s15.min()) for sw in sweeps)
    if sweep_var < 1e-3:
        mant, expo = f"{sweep_var:.1e}".split("e")
        L["@@SWEEPVAR@@"] = f"${mant}\\times 10^{{{int(expo)}}}$"
        P["@@SWEEPVAR@@"] = f"{mant}e{int(expo)}"
    else:
        L["@@SWEEPVAR@@"] = f"${sweep_var:.4f}$"
        P["@@SWEEPVAR@@"] = f"{sweep_var:.4f}"
    print(f"sweep variation across full p range: {sweep_var:.2e}")

    # r(p) medians from run A (recorded for the artifact; quoted in V-E text)
    pcA = A["per_candidate"][A["per_candidate"].tag != "selected"]
    base = A["prop1_epsilon"].set_index("case")

    def solver_pct_arr(xs, p):
        xs = sorted(xs)
        i = max(0, min(len(xs) - 1, int(len(xs) * p)))
        return xs[i]

    def r_of_p(p):
        rs = []
        for cid, grp in pcA.groupby("case"):
            H0 = solver_pct_arr(grp.hpwl_eval.values, p)
            A0 = solver_pct_arr(grp.area_eval.values, p)
            kh = H0 / base.loc[cid, "hpwl_star"]
            ka = A0 / base.loc[cid, "area_star"]
            rs.append(kh / ka)
        return np.array(rs)

    rp = {p: float(np.median(r_of_p(p)))
          for p in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)}
    print("r(p) medians:", {k: round(v, 2) for k, v in rp.items()})

    # ---------- figures
    figs = {}
    # calibration boxplot
    def boxstats(s):
        return dict(lw=np.percentile(s, 5), lq=np.percentile(s, 25),
                    med=np.median(s), uq=np.percentile(s, 75),
                    uw=np.percentile(s, 95))

    groups = [
        ("raw, DiT", g("DiT", "rho_raw"), "orange"),
        ("raw, classical", g("classical", "rho_raw"), "blue"),
        ("gap, DiT", g("DiT", "rho_gap"), "orange"),
        ("gap, classical", g("classical", "rho_gap"), "blue"),
    ]
    parts = []
    parts.append(r"""\begin{tikzpicture}
\begin{axis}[
  width=\columnwidth, height=58mm,
  boxplot/draw direction=y,
  xtick={1,2,3,4},
  xticklabels={DiT, classical, DiT, classical},
  xticklabel style={font=\scriptsize},
  yticklabel style={font=\scriptsize},
  ylabel={$\rho$ (normalised rank ratio)},
  label style={font=\scriptsize},
  ymin=0.45, ymax=2.0,
  grid=major,
  every axis plot/.append style={line width=0.7pt},
]""")
    for i, (lab, s, col) in enumerate(groups):
        st = boxstats(s)
        parts.append(
            f"\\addplot+[boxplot prepared={{lower whisker={st['lw']:.3f}, "
            f"lower quartile={st['lq']:.3f}, median={st['med']:.3f}, "
            f"upper quartile={st['uq']:.3f}, upper whisker={st['uw']:.3f}}}, "
            f"color={col}!80!black, fill={col}!20] coordinates {{}};")
    parts.append(r"""\draw[dashed, black!60, thick] (axis cs:0.5,1) -- (axis cs:4.5,1);
\node[font=\scriptsize, anchor=north east, fill=white, inner sep=1pt] at (axis cs:4.45,0.985) {calibration target};
\draw[black!50] (axis cs:2.5,0.45) -- (axis cs:2.5,2.0);
\node[font=\scriptsize\bfseries, anchor=north] at (axis cs:1.5,1.97) {raw cost $\widetilde{C}$ \eqref{eq:raw-rerank}};
\node[font=\scriptsize\bfseries, anchor=north] at (axis cs:3.5,1.97) {gap cost $\widehat{C}$ \eqref{eq:gap-rerank}};
\end{axis}
\end{tikzpicture}""")
    figs["@@CALIBRATIONFIGURE@@"] = "\n".join(parts)

    # sweep figure: groupplot with two stacked axes
    def line(coords):
        return " ".join(f"({p:.3f},{v:.5f})" for p, v in coords)

    sweep_parts = []
    sweep_parts.append(r"""\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=1 by 2, vertical sep=9mm},
  width=\columnwidth, height=44mm,
  xmin=0, xmax=1,
  label style={font=\scriptsize},
  xticklabel style={font=\scriptsize},
  yticklabel style={font=\scriptsize},
  legend style={font=\tiny, fill=white, draw=black!30},
  grid=major,
]""")
    # S15 axis
    sweep_parts.append(r"\nextgroupplot[ylabel={$S_{15}$}, legend pos=north west]")
    colors = ["blue", "red!80!black", "green!50!black"]
    for i, sw in enumerate(sweeps):
        sweep_parts.append(
            f"\\addplot+[mark=none, thick, color={colors[i]}] coordinates {{"
            + line(zip(sw.p, sw.s15)) + "};")
        sweep_parts.append(f"\\addlegendentry{{run {chr(65+i)}}}")
    s15_raw = A["summary"]["s15"]["raw"]
    sweep_parts.append(
        f"\\addplot+[mark=none, dotted, thick, black] coordinates {{(0,{s15_raw:.5f}) (1,{s15_raw:.5f})}};")
    sweep_parts.append("\\addlegendentry{raw policy (run A)}")
    sweep_parts.append(
        r"\draw[dashed, black!70] (axis cs:0.3,\pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:0.3,\pgfkeysvalueof{/pgfplots/ymax});")
    # S100 axis
    sweep_parts.append(r"\nextgroupplot[ylabel={$S_{100}$}, xlabel={rerank percentile $p$}]")
    for i, sw in enumerate(sweeps):
        sweep_parts.append(
            f"\\addplot+[mark=none, thick, color={colors[i]}] coordinates {{"
            + line(zip(sw.p, sw.s100)) + "};")
    s100_raw = raws[0]
    sweep_parts.append(
        f"\\addplot+[mark=none, dotted, thick, black] coordinates {{(0,{s100_raw:.5f}) (1,{s100_raw:.5f})}};")
    sweep_parts.append(
        r"\draw[dashed, black!70] (axis cs:0.3,\pgfkeysvalueof{/pgfplots/ymin}) -- (axis cs:0.3,\pgfkeysvalueof{/pgfplots/ymax});")
    sweep_parts.append(r"""\end{groupplot}
\end{tikzpicture}""")
    figs["@@SWEEPFIGURE@@"] = "\n".join(sweep_parts)

    # same-pool table (runs A-C gap-aware + run E raw end-to-end)
    rows = []
    for i, (rr, gg, oo) in enumerate(zip(raws, gaps, oras)):
        rows.append((chr(65 + i), rr, gg, oo))
    if "runE_raw" in runs:
        E = runs["runE_raw"]["summary"]["s100"]
        rows.append(("E", E["raw"], E["gap_030"], E["oracle"]))
    tbl = [r"\begin{tabular}{@{}lcccc@{}}",
           r"\toprule",
           r"Pool & Raw~\eqref{eq:raw-rerank} & Gap-aware~\eqref{eq:gap-rerank} & Oracle & Headroom recovered\\",
           r"\midrule"]
    for name, rr, gg, oo in rows:
        rec = (rr - gg) / (rr - oo) * 100 if rr > oo else float("nan")
        tbl.append(f"Run {name} & ${rr:.5f}$ & ${gg:.5f}$ & ${oo:.5f}$ & ${rec:.2f}\\%$\\\\")
    tbl += [r"\midrule",
            (f"\\multicolumn{{5}}{{@{{}}p{{0.95\\columnwidth}}@{{}}}}{{Paired statistics (run A pools): "
             f"gap better on {nw}, tie {nt}, worse {nl} of 100 instances; "
             f"Wilcoxon {L['@@WILCOXSTATS@@']}; "
             f"$\\Delta S_{{100}} = {fmt(boot['mean'],4)}$, "
             f"$95\\%$ CI ${L['@@BOOTCI@@']}$.}}\\\\"),
            r"\bottomrule", r"\end{tabular}"]
    figs["@@SAMEPOOLTABLE@@"] = "\n".join(tbl)

    # docx-only references to figure/table numbers.
    # NB: verify against the compiled paper.aux in the final pass.
    P["@@TABSAMEPOOL@@"] = "3"
    P["@@FIGCAL@@"] = "3"
    P["@@FIGSWEEP@@"] = "4"
    P["@@TABCROSS@@"] = "7"

    (PAPER_DIR / "placeholders_latex.json").write_text(
        json.dumps({**L, **{k: v for k, v in figs.items()}}, indent=1))
    (PAPER_DIR / "placeholders_plain.json").write_text(
        json.dumps(P, indent=1), encoding="utf-8")
    print("placeholders written:", len(L) + len(figs), "latex,", len(P), "plain")

    if args.fill:
        tmpl = PAPER_DIR / "paper_template.tex"
        if not tmpl.exists():
            raise SystemExit("snapshot paper.tex -> paper_template.tex first")
        text = tmpl.read_text(encoding="utf-8")
        allmap = {**L, **figs}
        import re as _re
        missing = set(_re.findall(r"@@[A-Z0-9]+@@", text)) - set(allmap)
        if missing:
            raise SystemExit(f"unresolved placeholders: {missing}")
        for k, v in allmap.items():
            text = text.replace(k, v)
        (PAPER_DIR / "paper.tex").write_text(text, encoding="utf-8")
        print("paper.tex filled from template")


if __name__ == "__main__":
    main()
