"""Figures for the paper from the exported CSVs.
python make_figs.py <series.csv> <allruns.csv> <outdir> [sens_allruns.csv]"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = pd.read_csv(sys.argv[1])
R = pd.read_csv(sys.argv[2])
out = sys.argv[3]
os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
# Okabe-Ito, colour-blind safe; marker redundancy so no information is colour-only
STY = {
    "Mondrian-CP": dict(c="#999999", m="s", ls="--", lab="Mondrian-CP (static)"),
    "ACI (all labels)": dict(c="#E69F00", m="^", ls="-.", lab="ACI, all labels (oracle)"),
    "Mondrian + random-label refresh": dict(c="#56B4E9", m="o", ls=":", lab="Random-label refresh (2x effort)"),
    "FACT (proposed)": dict(c="#0072B2", m="D", ls="-", lab="FACT (proposed)"),
}
CELLS = [("ciciot2023", "novel", "CICIoT2023, novel attacks"), ("nf_ton_v2", "novel", "NF-ToN-IoT-v2, novel attacks"),
         ("edgeiiot", "novel", "Edge-IIoTset, novel attacks"), ("iot23", "capture", "IoT-23, chronological captures")]

fig, ax = plt.subplots(4, 2, figsize=(7.0, 8.2), sharex="row")
for i, (d, sc, title) in enumerate(CELLS):
    for me, st in STY.items():
        g = S[(S.dataset == d) & (S.scenario == sc) & (S.method == me)].sort_values("flows")
        if g.empty:
            continue
        x = g.flows / 1e6
        kw = dict(color=st["c"], ls=st["ls"], lw=1.4, marker=st["m"], ms=3, markevery=6, label=st["lab"])
        ax[i, 0].plot(x, g["cov"], **kw)
        ax[i, 1].plot(x, g["err"], **kw)
    ax[i, 0].axhline(0.95, color="k", lw=0.6, ls=(0, (1, 2)))
    ax[i, 0].set_ylabel("coverage")
    ax[i, 1].set_ylabel("decision error")
    ax[i, 0].set_title(title, loc="left", fontsize=8)
    if sc == "novel":
        for a in ax[i]:
            a.axvspan(0.25, x.max() if len(x) else 1, color="#000000", alpha=0.04, lw=0)
    ax[i, 0].set_ylim(0.0 if d == "iot23" else 0.6, 1.02)
ax[3, 0].set_xlabel("flows processed (millions)")
ax[3, 1].set_xlabel("flows processed (millions)")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.005))
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{out}/fig_series.pdf")
plt.close(fig)

# worst-class coverage and decision error across all eight cells
M = R[(R.tag == "v2") | ((R.tag == "g") & (R.method == "FACT-G (rare-class guard)"))] if "tag" in R else R
order = [("ciciot2023", "novel"), ("ciciot2023", "capture"), ("ciciot2023", "prior"), ("nf_ton_v2", "novel"),
         ("nf_ton_v2", "prior"), ("edgeiiot", "novel"), ("edgeiiot", "prior"), ("iot23", "capture")]
names = ["CIC\nnovel", "CIC\ncapture", "CIC\nbursts", "ToN\nnovel", "ToN\nbursts", "Edge\nnovel", "Edge\nbursts",
         "IoT-23\ncapture"]
meths = {"Split-CP (LAC)": ("#CC79A7", "v", "Split-CP"), "Weighted-CP": ("#D55E00", "<", "Weighted-CP"),
         "LabelShift-CP": ("#009E73", ">", "LabelShift-CP"), **{k: (v["c"], v["m"], v["lab"]) for k, v in STY.items()},
         "FACT-G (rare-class guard)": ("#000000", "*", "FACT-G (guard)")}
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
xs = np.arange(len(order))
w = 0.8 / len(meths)
for j, (me, (c, mk, lab)) in enumerate(meths.items()):
    for k, key in enumerate(["worst_class_cov", "e2e_error"]):
        vals, err = [], []
        for d, sc in order:
            g = M[(M.dataset == d) & (M.scenario == sc) & (M.method == me)][key]
            vals.append(g.mean())
            err.append(g.std() / np.sqrt(max(1, len(g))))
        ax[k].errorbar(xs - 0.4 + w * (j + 0.5), vals, yerr=err, fmt=mk, color=c, ms=3.5, lw=0.8, capsize=1.5,
                       label=lab)
for k, (yl, ttl) in enumerate([("worst-class coverage", "(a) Worst-class coverage (higher is better)"),
                               ("end-to-end decision error", "(b) Decision error (lower is better)")]):
    ax[k].set_xticks(xs)
    ax[k].set_xticklabels(names, fontsize=6.5)
    ax[k].set_ylabel(yl)
    ax[k].set_title(ttl, loc="left", fontsize=8)
ax[0].axhline(0.95, color="k", lw=0.6, ls=(0, (1, 2)))
ax[1].set_yscale("log")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02), fontsize=6.5)
fig.tight_layout(rect=(0, 0, 1, 0.84))
fig.savefig(f"{out}/fig_cells.pdf")
plt.close(fig)

# sensitivity
if len(sys.argv) > 4 and os.path.exists(sys.argv[4]):
    Z = pd.read_csv(sys.argv[4])
    Z["v2"] = Z.tag.str.endswith("v2")
    Z = Z[Z.v2 | (Z.tag == "v2")].copy()
    Z["base"] = Z.tag.str.replace("v2$", "", regex=True)
    grids = [("capacity $\\kappa$", {"cap0.02": 0.02, "cap0.05": 0.05, "": 0.10, "cap0.2": 0.20}),
             ("audit rate $\\varepsilon$", {"eps0.005": 0.005, "eps0.01": 0.01, "": 0.02, "eps0.05": 0.05}),
             ("label delay (batches)", {"delay1": 1, "": 5, "delay20": 20}),
             ("analyst label noise", {"": 0.0, "noise0.05": 0.05, "noise0.1": 0.10})]
    cells3 = [("ciciot2023", "novel", "CIC novel"), ("iot23", "capture", "IoT-23"), ("nf_ton_v2", "novel", "ToN novel")]
    fig, ax = plt.subplots(2, 4, figsize=(7.0, 3.6))
    for gi, (xl, mp) in enumerate(grids):
        for (d, sc, lab), col in zip(cells3, ["#0072B2", "#D55E00", "#009E73"]):
            for row, key in enumerate(["worst_class_cov", "e2e_error"]):
                xv, yv = [], []
                for t, xval in mp.items():
                    g = Z[(Z.dataset == d) & (Z.scenario == sc) & (Z.base == t) & (Z.method == "FACT (proposed)")]
                    if t == "":
                        g = Z[(Z.dataset == d) & (Z.scenario == sc) & (Z.tag == "v2") & (Z.method == "FACT (proposed)")]
                    if len(g):
                        xv.append(xval)
                        yv.append(g[key].mean())
                ax[row, gi].plot(xv, yv, marker="o", ms=3, lw=1.2, color=col, label=lab)
        ax[1, gi].set_xlabel(xl)
        if gi in (0, 1):
            for r_ in (0, 1):
                ax[r_, gi].set_xscale("log")
    ax[0, 0].set_ylabel("worst-class coverage")
    ax[1, 0].set_ylabel("decision error")
    for a in ax[0]:
        a.axhline(0.95, color="k", lw=0.6, ls=(0, (1, 2)))
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{out}/fig_sens.pdf")
    plt.close(fig)
print("figures written to", out)
