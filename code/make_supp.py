"""Supplementary tables (all methods, all cells, with standard errors; full sensitivity grid).
python make_supp.py <table_all.csv> <outfile.tex>"""
import sys
import numpy as np
import pandas as pd

T = pd.read_csv(sys.argv[1])
NM = {"ciciot2023": "CICIoT2023", "nf_ton_v2": "NF-ToN-IoT-v2", "edgeiiot": "Edge-IIoTset", "iot23": "IoT-23"}
SC = {"novel": "novel attacks", "capture": "capture drift", "prior": "attack bursts"}
CELLS = [("ciciot2023", "novel"), ("ciciot2023", "capture"), ("ciciot2023", "prior"), ("nf_ton_v2", "novel"),
         ("nf_ton_v2", "prior"), ("edgeiiot", "novel"), ("edgeiiot", "prior"), ("iot23", "capture")]
ORDER = ["Split-CP (LAC)", "Mondrian-CP", "APS", "RAPS", "Weighted-CP", "LabelShift-CP", "ACI (all labels)",
         "ACI (analyst labels)", "Max-prob selective", "Mondrian + random-label refresh", "FACT (proposed)",
         "FACT-G (rare-class guard)", "FACT (all labels)", "FACT w/o IPW", "FACT w/o adaptive levels",
         "FACT w/o refresh"]


def pm(r, k, scale=1, f="%.3f"):
    mu, se = r[k], r.get(k + "_se", np.nan)
    if pd.isna(mu):
        return "--"
    return (f % (mu * scale)) + ("" if pd.isna(se) else "\\,$\\pm$\\," + (f % (se * scale)))


def esc(s):
    return s.replace("&", "\\&").replace("%", "\\%")


out = [r"""\documentclass[10pt]{article}
\usepackage[margin=2cm,landscape]{geometry}
\usepackage{booktabs,longtable}
\title{Supplementary information for ``Closing the Analyst Loop: Feedback-Adaptive
Conformal Triage for IoT Intrusion Detection under Distribution Shift''}
\author{Bambang Sugiantoro and Faiz Akhmad Hafizuddin}
\date{}
\begin{document}
\maketitle
\section*{S1. All methods in all cells}
Means $\pm$ standard errors over five seeds (FACT-G: separate array with the same seeds).
Cov.: overall coverage; worst: worst-class coverage (classes with at least 50 stream flows);
novel: coverage of classes absent from the source data; err.: end-to-end decision error (\%);
miss: attack miss rate among automatic decisions; effort: analyst effort (\%); set: mean set size;
demand: share of non-singleton sets (\%).
"""]
for d, sc in CELLS:
    out.append("\\subsection*{%s, %s}" % (NM[d], SC[sc]))
    out.append("\\begin{longtable}{lrrrrrrrr}\\toprule")
    out.append("Method & cov. & worst & novel & err. (\\%) & miss & effort (\\%) & set & demand (\\%)\\\\\\midrule")
    for me in ORDER:
        tag = "g2" if me.startswith("FACT-G") else "v2"
        g = T[(T.tag == tag) & (T.dataset == d) & (T.scenario == sc) & (T.method == me)]
        if g.empty:
            continue
        r = g.iloc[0].to_dict()
        out.append(" & ".join([esc(me), pm(r, "coverage"), pm(r, "worst_class_cov", f="%.2f"),
                               pm(r, "novel_cov", f="%.2f"), pm(r, "e2e_error", 100, "%.2f"),
                               pm(r, "attack_miss", f="%.3f"), pm(r, "human", 100, "%.1f"),
                               pm(r, "mean_set", f="%.2f"), pm(r, "demand", 100, "%.1f")]) + "\\\\")
    out.append("\\bottomrule\\end{longtable}")

out.append("\\section*{S2. Sensitivity grid (FACT, three seeds)}")
LAB = {"v2": "default", "cap0.02v2": "$\\kappa=2\\%$", "cap0.05v2": "$\\kappa=5\\%$", "cap0.2v2": "$\\kappa=20\\%$",
       "eps0.005v2": "$\\varepsilon=0.5\\%$", "eps0.01v2": "$\\varepsilon=1\\%$", "eps0.05v2": "$\\varepsilon=5\\%$",
       "delay1v2": "$\\Delta=1$", "delay20v2": "$\\Delta=20$", "alpha0.1v2": "$\\alpha=0.1$",
       "noise0.05v2": "noise 5\\%", "noise0.1v2": "noise 10\\%", "gam0.01v2": "$\\gamma=0.01$",
       "gam0.2v2": "$\\gamma=0.2$"}
for d, sc in [("ciciot2023", "novel"), ("nf_ton_v2", "novel"), ("iot23", "capture")]:
    out.append("\\subsection*{%s, %s}" % (NM[d], SC[sc]))
    out.append("\\begin{longtable}{lrrrrr}\\toprule")
    out.append("Configuration & cov. & worst & err. (\\%) & effort (\\%) & set\\\\\\midrule")
    for tag, lab in LAB.items():
        g = T[(T.tag == tag) & (T.dataset == d) & (T.scenario == sc) & (T.method == "FACT (proposed)")]
        if g.empty:
            continue
        r = g.iloc[0].to_dict()
        if tag == "v2":
            lab += " (five seeds)"
        out.append(" & ".join([lab, pm(r, "coverage"), pm(r, "worst_class_cov", f="%.2f"),
                               pm(r, "e2e_error", 100, "%.2f"), pm(r, "human", 100, "%.1f"),
                               pm(r, "mean_set", f="%.2f")]) + "\\\\")
    out.append("\\bottomrule\\end{longtable}")
out.append("\\end{document}")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("written", sys.argv[2])
