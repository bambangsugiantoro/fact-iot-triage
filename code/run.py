"""One (dataset, scenario, seed) cell: build source/stream, train the base
detector once, run every method, write results/<dataset>_<scenario>_s<seed>.json.

Scenarios
  novel   : unseen attack types emerge mid-stream (gradual ramp after 25% of the stream)
  capture : train on early captures/files, deploy on later ones (natural covariate + label shift)
  prior   : in-distribution flows whose class mix drifts in bursts (pure label shift)
Usage: python run.py <dataset> <scenario> <seed> [--capacity 0.1] [--alpha 0.05] [--methods all|core]
"""
import argparse, copy, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (Detector, ETDetector, SplitLAC, Mondrian, APS, RAPS, WeightedCP, LabelShiftCP, ACI, MaxProb,
                  RefreshRandom, FACT, run_stream)

ROOT = os.environ.get("IOT_ROOT", "/virtiofs/work/bambang/iot_exp")
BENIGN = {"Benign", "Normal", "benign"}
NOVEL = {
    "ciciot2023": ["DNS_Spoofing", "MITM-ArpSpoofing", "XSS", "SqlInjection", "CommandInjection",
                   "BrowserHijacking", "Backdoor_Malware", "Uploading_Attack", "DictionaryBruteForce"],
    "nf_ton_v2": ["backdoor", "ransomware", "mitm", "xss"],
    "edgeiiot": ["Ransomware", "MITM", "Fingerprinting", "XSS", "Uploading"],
}

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("scenario")
ap.add_argument("seed", type=int)
ap.add_argument("--capacity", type=float, default=0.10)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--delay", type=int, default=5)
ap.add_argument("--eps", type=float, default=0.02)
ap.add_argument("--methods", default="all")
ap.add_argument("--stream_n", type=int, default=1_000_000)
ap.add_argument("--train_n", type=int, default=500_000)
ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--tag", default="")
ap.add_argument("--detector", default="lgbm", choices=["lgbm", "et"])
ap.add_argument("--label_noise", type=float, default=0.0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
t0 = time.time()

df = pd.read_parquet(f"{ROOT}/data/proc/{a.dataset}.parquet")
vc = df.y.value_counts()
df = df[df.y.isin(vc[vc >= 100].index)].reset_index(drop=True)  # drop classes with <100 flows in total
feats = [c for c in df.columns if c not in ("y", "grp", "ts")]
classes = sorted(df.y.unique())
cid = {c: i for i, c in enumerate(classes)}
K = len(classes)
y_all = df.y.map(cid).to_numpy()
X_all = df[feats].to_numpy(np.float32)
grp = df.grp.to_numpy()
ts = df.ts.to_numpy()
attack = np.array([c not in BENIGN for c in classes])
del df


def take(idx, n):
    return idx if len(idx) <= n else rng.choice(idx, n, replace=False)


def split_src(idx):
    idx = rng.permutation(idx)
    ncal = min(100_000, max(5_000, len(idx) // 5))
    cal, tr = idx[:ncal], idx[ncal:]
    return take(tr, a.train_n), cal


if a.scenario == "novel":
    H = np.isin(np.array(classes)[y_all], NOVEL[a.dataset])
    known = np.flatnonzero(~H)
    perm = rng.permutation(known)
    half = len(perm) // 2
    tr, cal = split_src(perm[:half])
    kn = take(perm[half:], int(a.stream_n * 0.85))
    nov = rng.permutation(take(np.flatnonzero(H), int(a.stream_n * 0.15)))
    # novel flows enter after the first 25% of the stream with a linearly increasing rate
    n = len(kn) + len(nov)
    w = np.linspace(0, 1, n)
    w[: n // 4] = 0
    pos = np.sort(rng.choice(n, len(nov), replace=False, p=w / w.sum()))
    order = np.empty(n, np.int64)
    mask = np.zeros(n, bool)
    mask[pos] = True
    order[mask] = nov
    order[~mask] = rng.permutation(kn)
    stream_idx = order
elif a.scenario == "capture":
    # per class: earliest 40% of its captures/files (or rows) = source; the rest = stream (in time order)
    src_i, str_i = [], []
    for c in range(K):
        ii = np.flatnonzero(y_all == c)
        ii = ii[np.lexsort((ts[ii], grp[ii]))]
        g = np.unique(grp[ii])
        if a.dataset == "iot23":
            continue
        if len(g) >= 3:
            cut = g[int(np.floor(0.4 * len(g)))]
            src_i.append(ii[grp[ii] < cut])
            str_i.append(ii[grp[ii] >= cut])
        else:
            k = int(0.4 * len(ii))
            src_i.append(ii[:k])
            str_i.append(ii[k:])
    if a.dataset == "iot23":
        # chronological captures: earliest captures holding ~40% of the flows are the source
        order_c = np.argsort([ts[grp == gg].min() for gg in np.unique(grp)])
        gs = np.unique(grp)[order_c]
        cnt = np.array([np.sum(grp == gg) for gg in gs])
        k = np.searchsorted(np.cumsum(cnt) / cnt.sum(), 0.4) + 1
        srcg = gs[:k]
        sm = np.isin(grp, srcg)
        src_i = [np.flatnonzero(sm)]
        st = np.flatnonzero(~sm)
        str_i = [st[np.argsort(ts[st])]]
    src_all = np.concatenate(src_i)
    tr, cal = split_src(src_all)
    st = np.concatenate(str_i)
    if a.dataset != "iot23":
        # interleave captures across classes: sort by relative capture position
        rel = np.zeros(len(st))
        for c in range(K):
            m = y_all[st] == c
            if m.any():
                r = np.argsort(np.argsort(np.lexsort((ts[st[m]], grp[st[m]]))))
                rel[m] = r / max(1, m.sum() - 1)
        st = st[np.argsort(rel + 1e-6 * rng.random(len(st)))]
    if len(st) > a.stream_n:  # keep temporal order while thinning
        keep = np.sort(rng.choice(len(st), a.stream_n, replace=False))
        st = st[keep]
    stream_idx = st
elif a.scenario == "prior":
    perm = rng.permutation(len(y_all))
    half = len(perm) // 2
    tr, cal = split_src(perm[:half])
    pool = perm[half:]
    byc = {c: list(rng.permutation(pool[y_all[pool] == c])) for c in range(K)}
    seg, out = 50_000, []
    att_ids = [c for c in range(K) if attack[c]]
    ben_ids = [c for c in range(K) if not attack[c]]
    while len(out) < a.stream_n:
        mix = np.zeros(K)
        pb = rng.uniform(0.05, 0.9)
        mix[att_ids] = (1 - pb) * rng.dirichlet(np.full(len(att_ids), 0.3))
        mix[ben_ids] = pb / max(1, len(ben_ids))
        cnts = rng.multinomial(seg, mix)
        took = 0
        for c, m in enumerate(cnts):
            m = min(m, len(byc[c]))
            out.extend(byc[c][:m])
            byc[c] = byc[c][m:]
            took += m
        if took == 0:
            break
    # shuffle within segments only
    out = np.array(out[: a.stream_n])
    for s in range(0, len(out), seg):
        out[s:s + seg] = rng.permutation(out[s:s + seg])
    stream_idx = out
else:
    raise SystemExit("unknown scenario")

src = dict(Xtr=X_all[tr], ytr=y_all[tr], Xcal=X_all[cal], ycal=y_all[cal])
stream = dict(X=X_all[stream_idx], y=y_all[stream_idx])
info = dict(dataset=a.dataset, scenario=a.scenario, seed=a.seed, K=K, classes=classes,
            attack=attack.tolist(), n_features=len(feats), n_train=len(tr), n_cal=len(cal),
            n_stream=len(stream_idx), src_classes=sorted(set(map(int, np.unique(src["ytr"])))),
            stream_class_counts={int(c): int(v) for c, v in zip(*np.unique(stream["y"], return_counts=True))},
            capacity=a.capacity, alpha=a.alpha, delay=a.delay, eps=a.eps, detector=a.detector,
            label_noise=a.label_noise)
del X_all
print(json.dumps({k: v for k, v in info.items() if k not in ("classes", "stream_class_counts")}), flush=True)

DET = ETDetector if a.detector == "et" else Detector
base = DET(K, threads=a.threads, rounds=150, leaves=63, seed=a.seed)
tt = time.time()
base.fit(src["Xtr"], src["ytr"])
info["train_s"] = time.time() - tt
tt = time.time()
P_all = base.proba(stream["X"])
info["proba_s_per_Mflow"] = (time.time() - tt) / len(stream_idx) * 1e6
pred = P_all.argmax(1)
info["static_acc_stream"] = float((pred == stream["y"]).mean())
Pc = base.proba(src["Xcal"])
info["static_acc_cal"] = float((Pc.argmax(1) == src["ycal"]).mean())
print("base detector: train %.0fs, cal acc %.4f, stream acc %.4f" % (info["train_s"], info["static_acc_cal"],
                                                                      info["static_acc_stream"]), flush=True)

al = a.alpha
core = [Mondrian(al), ACI(al, True), RefreshRandom(al), FACT(al, eps=a.eps)]
allm = [SplitLAC(al), Mondrian(al), APS(al), RAPS(al), WeightedCP(al), LabelShiftCP(al), MaxProb(al),
        ACI(al, True), ACI(al, False), RefreshRandom(al), FACT(al, eps=a.eps),
        FACT(al, eps=a.eps, name="FACT (all labels)"),
        FACT(al, eps=a.eps, ipw=False, name="FACT w/o IPW"),
        FACT(al, eps=a.eps, aci=False, name="FACT w/o adaptive levels"),
        FACT(al, eps=a.eps, refresh=False, name="FACT w/o refresh")]
allm[11].oracle_labels = True  # FACT (all labels): every label revealed, an upper reference
methods = core if a.methods == "core" else allm
results = []
for M in methods:
    det = copy.deepcopy(base) if M.refreshes else base
    r = run_stream(M, det, src, stream, K, attack, alpha=al, capacity=a.capacity, delay=a.delay,
                   seed=a.seed, P_all=P_all, label_noise=a.label_noise)
    print("%-32s cov=%.3f worst=%.3f auto=%.3f acc=%.4f load=%.3f e2e=%.5f miss=%.4f set=%.2f %.0fs" % (
        r["method"], r["coverage"], r["worst_class_cov"], r["automation"], r["auto_acc"], r["analyst_load"],
        r["e2e_error"], r["attack_miss"], r["mean_set"], r["wall_s"]), flush=True)
    results.append(r)
info["total_s"] = time.time() - t0
os.makedirs(f"{ROOT}/results", exist_ok=True)
out = f"{ROOT}/results/{a.dataset}_{a.scenario}_s{a.seed}{a.tag}.json"
json.dump(dict(info=info, results=results), open(out, "w"))
print("saved", out, "total %.0fs" % info["total_s"], flush=True)
