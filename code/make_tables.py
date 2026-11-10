"""Paper tables and number macros from the exported CSVs.
python make_tables.py <table_all.csv> <all_runs.csv> <all_info.csv> <outdir>"""
import sys, os
import numpy as np
import pandas as pd
from scipy import stats

T = pd.read_csv(sys.argv[1])
R = pd.read_csv(sys.argv[2])
I = pd.read_csv(sys.argv[3])
out = sys.argv[4]
os.makedirs(out, exist_ok=True)

MAIN, GUARD = "v2", "g2"
CELLS = [("ciciot2023", "novel"), ("ciciot2023", "capture"), ("ciciot2023", "prior"), ("nf_ton_v2", "novel"),
         ("nf_ton_v2", "prior"), ("edgeiiot", "novel"), ("edgeiiot", "prior"), ("iot23", "capture")]
HEAD = ["\\multicolumn{3}{c}{CICIoT2023}", "\\multicolumn{2}{c}{NF-ToN-IoT-v2}", "\\multicolumn{2}{c}{Edge-IIoTset}",
        "IoT-23"]
SUB = "& novel & capture & bursts & novel & bursts & novel & bursts & capture"
METH = [("Split-CP (LAC)", "Split CP (LAC)"), ("Mondrian-CP", "Mondrian CP"), ("APS", "APS"), ("RAPS", "RAPS"),
        ("Weighted-CP", "Weighted CP"), ("LabelShift-CP", "Label-shift CP"),
        ("ACI (all labels)", "ACI, all labels$^{\\ast}$"), ("ACI (analyst labels)", "ACI, analyst labels"),
        ("Max-prob selective", "Max-prob selective"),
        ("Mondrian + random-label refresh", "Random-label refresh$^{\\dagger}$"),
        ("FACT (proposed)", "\\textbf{FACT}"), ("FACT-G (rare-class guard)", "\\textbf{FACT-G}")]


def runs(method, d, sc, key):
    tag = GUARD if method.startswith("FACT-G") else MAIN
    g = R[(R.tag == tag) & (R.dataset == d) & (R.scenario == sc) & (R.method == method)]
    return g.sort_values("seed")[key].to_numpy(float)


def cellval(method, d, sc, key):
    v = runs(method, d, sc, key)
    return (np.nan, np.nan, 0) if len(v) == 0 else (v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, len(v))


def table(key, scale, fmt, best, caption, label, fname):
    vals = {m: [cellval(m, d, sc, key) for d, sc in CELLS] for m, _ in METH}
    lines = ["\\begin{table*}[t]", "\\centering", "\\caption{" + caption + "}", "\\label{" + label + "}",
             "\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{l" + "r" * len(CELLS) + "}",
             "\\toprule", "Method & " + " & ".join(HEAD) + " \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-9}", SUB + " \\\\", "\\midrule"]
    for j in range(len(CELLS)):
        col = [vals[m][j][0] for m, _ in METH if not m.startswith("ACI (all")]
        col = [c for c in col if not np.isnan(c)]
        vals.setdefault("_best", []).append((max(col) if best == "max" else min(col)) if col else np.nan)
    for m, lab in METH:
        if m == "Mondrian + random-label refresh":
            lines.append("\\midrule")
        cells = []
        for j, (mu, se, n) in enumerate(vals[m]):
            if np.isnan(mu):
                cells.append("--")
                continue
            s = fmt % (mu * scale)
            if not m.startswith("ACI (all") and abs(mu - vals["_best"][j]) < 1e-9:
                s = "\\textbf{" + s + "}"
            cells.append(s)
        lines.append(lab + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}"]
    open(f"{out}/{fname}", "w").write("\n".join(lines) + "\n")
    return vals


Wv = table("worst_class_cov", 1, "%.2f", "max",
           "Worst-class coverage (minimum over classes with at least 50 stream flows; target $0.95$), mean over "
           "five seeds. $^{\\ast}$Oracle that observes every label. $^{\\dagger}$Uses about twice the human effort "
           "of FACT (Table~\\ref{tab:effort}). Best non-oracle value per column in bold; standard errors are in "
           "the supplementary material.", "tab:worst", "tab_worst.tex")
Ev = table("e2e_error", 100, "%.2f", "min",
           "End-to-end decision error (\\%): flows decided automatically and wrongly, over all flows, at the same "
           "analyst capacity $\\kappa=10\\%$. Mean over five seeds; notation as in Table~\\ref{tab:worst}.",
           "tab:err", "tab_err.tex")

# ---- ablations / oracle on the four shift cells
AB = [("FACT (proposed)", "FACT"), ("FACT-G (rare-class guard)", "FACT-G (rare-class guard)"),
      ("FACT (all labels)", "FACT, all labels (oracle)"), ("FACT w/o IPW", "w/o IPW"),
      ("FACT w/o adaptive levels", "w/o adaptive levels"), ("FACT w/o refresh", "w/o refresh"),
      ("Mondrian-CP", "Mondrian CP (static)")]
SH = [("ciciot2023", "novel", "CIC novel"), ("nf_ton_v2", "novel", "ToN novel"), ("edgeiiot", "novel", "Edge novel"),
      ("iot23", "capture", "IoT-23 capture"), ("edgeiiot", "prior", "Edge bursts")]
L = ["\\begin{table*}[t]", "\\centering",
     "\\caption{Ablations and the cost of selective labels. Worst: worst-class coverage; set: mean set size; "
     "err.: decision error (\\%). Mean over five seeds. Overall coverage is in the supplementary material.}", "\\label{tab:ablation}",
     "\\begin{adjustbox}{max width=\\textwidth}", "\\begin{tabular}{l" + "rrr" * len(SH) + "}", "\\toprule",
     "Variant & " + " & ".join("\\multicolumn{3}{c}{%s}" % s[2] for s in SH) + " \\\\",
     "".join("\\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i) for i in range(len(SH))),
     " & " + " & ".join(["worst & set & err."] * len(SH)) + " \\\\", "\\midrule"]
for m, lab in AB:
    row = []
    for d, sc, _ in SH:
        for k, sc_, f in [("worst_class_cov", 1, "%.2f"), ("mean_set", 1, "%.1f"),
                          ("e2e_error", 100, "%.2f")]:
            mu = cellval(m, d, sc, k)[0]
            row.append("--" if np.isnan(mu) else f % (mu * sc_))
    L.append(lab + " & " + " & ".join(row) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table*}"]
open(f"{out}/tab_ablation.tex", "w").write("\n".join(L) + "\n")

# ---- human effort and triage demand
EF = [("Mondrian-CP", "Mondrian CP"), ("Mondrian + random-label refresh", "Random-label refresh"),
      ("FACT w/o adaptive levels", "FACT w/o adaptive levels"), ("FACT (proposed)", "FACT")]
R["human"] = R["analyst_load"]
rr = R.method == "Mondrian + random-label refresh"
R.loc[rr, "human"] = 2 * R.loc[rr, "escalation"] + 0.02 * R.loc[rr, "automation"]
SH2 = [("ciciot2023", "novel", "CIC novel"), ("ciciot2023", "capture", "CIC capture"),
       ("nf_ton_v2", "novel", "ToN novel"), ("edgeiiot", "novel", "Edge novel"), ("iot23", "capture", "IoT-23")]
L = ["\\begin{table}[t]", "\\centering",
     "\\caption{Human effort (flows inspected or labelled by an analyst, \\% of the stream), triage demand (flows "
     "whose set is not a singleton, \\%) and decision error (\\%) on the shift cells. Mean over five seeds.}",
     "\\label{tab:effort}", "\\small",
     "\\begin{tabular}{ll" + "r" * 3 + "}", "\\toprule", "Cell & Method & effort & demand & err. \\\\", "\\midrule"]
for d, sc, nm in SH2:
    for i, (m, lab) in enumerate(EF):
        mu = [cellval(m, d, sc, k)[0] for k in ("human", "demand", "e2e_error")]
        L.append(("\\multirow{4}{*}{%s}" % nm if i == 0 else "") + " & " + lab + " & " +
                 " & ".join("--" if np.isnan(v) else "%.1f" % (100 * v) for v in mu[:2]) + " & " +
                 ("--" if np.isnan(mu[2]) else "%.2f" % (100 * mu[2])) + " \\\\")
    L.append("\\midrule" if (d, sc) != SH2[-1][:2] else "")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{out}/tab_effort.tex", "w").write("\n".join([x for x in L if x]) + "\n")

# ---- computational cost (main runs)
Im = I[I.tag == MAIN]
L = ["\\begin{table}[t]", "\\centering",
     "\\caption{Computational cost on one HPC node (16 CPU cores per run). Train: initial detector fit on "
     "$500$k flows; score: detector inference; FACT: per-flow cost of FACT including sets, feedback and "
     "refreshes; throughput: flows per second of the full FACT loop. Mean over seeds and scenarios.}",
     "\\label{tab:cost}", "\\small", "\\begin{tabular}{lrrrr}", "\\toprule",
     "Corpus & train (s) & score ($\\mu$s/flow) & FACT ($\\mu$s/flow) & flows/s \\\\", "\\midrule"]
NM = {"ciciot2023": "CICIoT2023", "nf_ton_v2": "NF-ToN-IoT-v2", "edgeiiot": "Edge-IIoTset", "iot23": "IoT-23"}
thr = {}
for d in NM:
    a = Im[Im.dataset == d]
    f = R[(R.tag == MAIN) & (R.dataset == d) & (R.method == "FACT (proposed)")]
    if a.empty or f.empty:
        continue
    us_fact = f.wall_s.mean() / a.n_stream.mean() * 1e6  # microseconds per flow, full FACT loop
    thr[d] = 1e6 / us_fact
    L.append("%s & %.1f & %.2f & %.1f & %s \\\\" % (NM[d], a.train_s.mean(), a.proba_s_per_Mflow.mean(), us_fact,
                                                    "{:,.0f}".format(thr[d]).replace(",", "{,}")))
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(f"{out}/tab_cost.tex", "w").write("\n".join(L) + "\n")

# ---- paired statistics FACT vs Mondrian (decision error, worst-class coverage)
st = []
for d, sc in CELLS:
    a = R[(R.tag == MAIN) & (R.dataset == d) & (R.scenario == sc)]
    f = a[a.method == "FACT (proposed)"].set_index("seed")
    m = a[a.method == "Mondrian-CP"].set_index("seed")
    s = f.index.intersection(m.index)
    if len(s) < 2:
        continue
    de = (f.loc[s, "e2e_error"] - m.loc[s, "e2e_error"]).to_numpy()
    dw = (f.loc[s, "worst_class_cov"] - m.loc[s, "worst_class_cov"]).to_numpy()
    t = stats.ttest_1samp(de, 0.0)
    ci = stats.t.ppf(0.975, len(de) - 1) * de.std(ddof=1) / np.sqrt(len(de))
    st.append(dict(dataset=d, scenario=sc, n=len(s), d_err=de.mean(), ci=ci, p=t.pvalue,
                   rel=de.mean() / m.loc[s, "e2e_error"].mean(), d_worst=dw.mean(), wins=int((de < 0).sum())))
ST = pd.DataFrame(st)
ST.to_csv(f"{out}/paired_stats.csv", index=False)
print(ST.round(4).to_string())

# ---- macros
def mac(name, val):
    return "\\newcommand{\\%s}{%s}\n" % (name, val)


nov = ST[ST.scenario != "prior"]
best = nov.rel.min()
txt = "% generated by make_tables.py -- do not edit\n"
txt += mac("FACTbestErrRed", "%.0f\\%%" % (-100 * best))
txt += mac("FACTthroughput", "{:,.0f}".format(min(thr.values()) if thr else 0).replace(",", "{,}"))
txt += mac("FACTthroughputMax", "{:,.0f}".format(max(thr.values()) if thr else 0).replace(",", "{,}"))
txt += mac("CODEURL", "https://github.com/bambangsugiantoro/fact-iot-triage")
nw = int(sum(ST[ST.scenario != "prior"].wins))
nt = int(sum(ST[ST.scenario != "prior"].n))
txt += mac("FACTwins", "%d of %d" % (nw, nt))
for _, r in ST.iterrows():
    k = {"ciciot2023": "Cic", "nf_ton_v2": "Ton", "edgeiiot": "Edge", "iot23": "Iot"}[r.dataset] + \
        {"novel": "Novel", "capture": "Capture", "prior": "Prior"}[r.scenario]
    txt += mac("Rel" + k, "%.0f\\%%" % (-100 * r.rel))
    txt += mac("P" + k, ("$p<0.001$" if r.p < 0.001 else "$p=%.3f$" % r.p))
open(f"{out}/numbers.tex", "w").write(txt)
print(txt)
