"""Export compact tables for the paper (run after aggregate.py)."""
import os
import numpy as np
import pandas as pd

ROOT = "/virtiofs/work/bambang/iot_exp"
os.chdir(ROOT)
R = pd.read_csv("results/all_runs.csv")
I = pd.read_csv("results/all_info.csv")
os.makedirs("results/paper", exist_ok=True)
R["human"] = R["analyst_load"]
rr = R["method"] == "Mondrian + random-label refresh"
# random-label control: analyst resolves its escalations AND labels as many random flows again
R.loc[rr, "human"] = 2 * R.loc[rr, "escalation"] + 0.02 * R.loc[rr, "automation"]
K = ["coverage", "worst_class_cov", "novel_cov", "auto_acc", "e2e_error", "attack_miss", "false_alarm",
     "analyst_load", "human", "mean_set", "infer_us_per_flow", "wall_s", "refreshes", "demand", "worst_class_silent"]
g = R.groupby(["tag", "dataset", "scenario", "method"])
m = g[K].mean()
s = g[K].std() / np.sqrt(g.size().values[:, None])
s.columns = [c + "_se" for c in s.columns]
out = m.join(s)
out["n"] = g.size()
out.reset_index().round(5).to_csv("results/paper/table_all.csv", index=False)
I.groupby(["dataset", "scenario"]).mean(numeric_only=True).round(4).to_csv("results/paper/info.csv")

# compact series: rolling metrics for the main figure
S = pd.read_csv("results/series.csv")
keep = ["Mondrian-CP", "LabelShift-CP", "ACI (all labels)", "Mondrian + random-label refresh", "FACT (proposed)",
        "FACT w/o IPW"]
S = S[S.method.isin(keep)]
rows = []
for (d, sc, me), gg in S.groupby(["dataset", "scenario", "method"]):
    gg = gg.sort_values("flows")
    idx = np.linspace(0, len(gg) - 1, min(40, len(gg))).round().astype(int)
    rows.append(gg.iloc[idx])
pd.concat(rows).round(4).to_csv("results/paper/series_compact.csv", index=False)
print("ok", out.shape)
