"""Aggregate result JSONs into compact CSVs (run on the HPC after the arrays finish)."""
import glob, json, os, re
import numpy as np
import pandas as pd

ROOT = "/virtiofs/work/bambang/iot_exp"
os.chdir(ROOT)
KEYS = ["coverage", "worst_class_cov", "mean_class_cov", "automation", "auto_acc", "escalation", "audit",
        "analyst_load", "e2e_error", "attack_miss", "false_alarm", "mean_set", "infer_us_per_flow",
        "refreshes", "wall_s"]
XK = ["demand", "worst_class_silent"]
rows, infos, series = [], [], []
for f in sorted(glob.glob("results/*.json")):
    base = os.path.basename(f)[:-5]
    m = re.match(r"(.+?)_(novel|capture|prior)_s(\d+)(?:_(.+))?$", base)
    if not m:
        continue
    ds, sc, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4) or "main"
    d = json.load(open(f))
    inf = d["info"]
    infos.append(dict(dataset=ds, scenario=sc, seed=seed, tag=tag, K=inf["K"], n_features=inf["n_features"],
                      n_train=inf["n_train"], n_cal=inf["n_cal"], n_stream=inf["n_stream"],
                      n_src_classes=len(inf["src_classes"]), train_s=inf["train_s"],
                      proba_s_per_Mflow=inf["proba_s_per_Mflow"], acc_cal=inf["static_acc_cal"],
                      acc_stream=inf["static_acc_stream"], total_s=inf["total_s"]))
    for r in d["results"]:
        row = dict(dataset=ds, scenario=sc, seed=seed, tag=tag, method=r["method"])
        row.update({k: r[k] for k in KEYS})
        row.update({k: r.get(k, np.nan) for k in XK})
        # classes never seen in training (novel) vs. seen
        src = set(inf["src_classes"])
        pc, pn = r["per_class_cov"], r["per_class_n"]
        nov = [int(c) for c in pc if int(c) not in src and pn[c] >= 50]
        row["novel_cov"] = float(np.mean([pc[str(c)] for c in nov])) if nov else np.nan
        rows.append(row)
        if tag == "v2" and seed in (0, 1, 2):
            for p in r["series"]:
                series.append(dict(dataset=ds, scenario=sc, seed=seed, method=r["method"], flows=p[0],
                                   auto=p[1], cov=p[2], err=p[3], esc=p[4]))
R = pd.DataFrame(rows)
I = pd.DataFrame(infos)
R.to_csv("results/all_runs.csv", index=False)
I.to_csv("results/all_info.csv", index=False)


def agg(df):
    g = df.groupby(["dataset", "scenario", "tag", "method"])
    out = g[KEYS + XK + ["novel_cov"]].mean()
    se = g[KEYS + XK + ["novel_cov"]].std() / np.sqrt(g.size().values[:, None])
    se.columns = [c + "_se" for c in se.columns]
    out = out.join(se)
    out["n_seeds"] = g.size()
    return out.reset_index()


agg(R).to_csv("results/summary.csv", index=False)
S = pd.DataFrame(series)
if len(S):
    # rolling means over 10 records, averaged over seeds, then thinned to ~60 points
    S = S.sort_values(["dataset", "scenario", "method", "seed", "flows"])
    for c in ["auto", "cov", "err", "esc"]:
        S[c] = S.groupby(["dataset", "scenario", "method", "seed"])[c].transform(
            lambda v: v.rolling(10, min_periods=1).mean())
    M = S.groupby(["dataset", "scenario", "method", "flows"])[["auto", "cov", "err", "esc"]].mean().reset_index()
    M = M[M.groupby(["dataset", "scenario", "method"]).cumcount() % 3 == 0]
    M.to_csv("results/series.csv", index=False)
print("runs", len(R), "cells", R.groupby(["dataset", "scenario", "tag"]).ngroups)
print(I.groupby(["dataset", "scenario"])[["K", "n_features", "n_train", "n_stream", "acc_cal", "acc_stream",
                                          "train_s", "total_s"]].mean().round(4).to_string())
