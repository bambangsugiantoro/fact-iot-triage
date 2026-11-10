"""Convert the raw public releases into data/proc/<dataset>.parquet
(float32 features + y label, grp capture/file index, ts start time).

Rare-preserving caps keep the experiments tractable while retaining every
flow of the rare classes that matter for per-class coverage:
  ciciot2023 : 34 classes from 309 per-capture CSVs (46.6M flows); at most
               400k flows per class, drawn uniformly across that class's files
  nf_ton_v2  : 16.9M flows, 10 classes; at most 500k flows per class
  edgeiiot   : all 1.91M flows, 15 classes (ML-ready release, 91 features)
  iot23      : 23 captures, 325M Zeek flows; per capture at most 150k flows
               per label (all flows of rarer labels), 22 flow features
Usage: python code/prep.py
"""
import glob, os, re, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.environ.get("IOT_ROOT", "/virtiofs/work/bambang/iot_exp")
os.chdir(ROOT)
os.makedirs("data/proc", exist_ok=True)
rng = np.random.default_rng(42)


def natkey(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def num(df, cols):
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").astype("float32") for c in cols})
    return X.replace([np.inf, -np.inf], np.nan)


def cap_per_class(D, cap):
    keep = []
    for c, g in D.groupby("y", sort=False):
        keep.append(g if len(g) <= cap else g.sample(cap, random_state=42))
    return pd.concat(keep, ignore_index=True)


def save(D, name):
    D.to_parquet(f"data/proc/{name}.parquet", index=False)
    print(name, D.shape, flush=True)
    print(D.y.value_counts().to_string(), flush=True)


t0 = time.time()
# ---------------- CIC-IoT-2023: one folder per class, one CSV per capture
parts, nraw = [], 0
for d in sorted(glob.glob("data/raw/ciciot2023/*/")):
    cls = os.path.basename(os.path.dirname(d))
    cls = "Benign" if cls.lower().startswith("benign") else cls
    files = sorted(glob.glob(d + "*.csv"), key=lambda p: natkey(os.path.basename(p)))
    tot = sum(sum(1 for _ in open(f)) - 1 for f in files)
    nraw += tot
    frac = min(1.0, 400_000 / max(tot, 1))
    for i, f in enumerate(files):
        df = pd.read_csv(f, low_memory=False)
        if frac < 1.0:
            df = df.sample(frac=frac, random_state=42)
        X = num(df, list(df.columns))
        X["y"], X["grp"], X["ts"] = cls, np.int32(i), np.float32(np.nan)
        parts.append(X)
D = pd.concat(parts, ignore_index=True)
print("ciciot2023 raw flows", nraw, "kept", len(D))
save(D, "ciciot2023")

# ---------------- NF-ToN-IoT-v2 (NetFlow v2, identifiers removed)
ID = {"IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "Label", "Attack"}
parts = []
for f in ["data/raw/nf_ton_v2/train.csv", "data/raw/nf_ton_v2/test.csv"]:
    for ch in pd.read_csv(f, chunksize=2_000_000, low_memory=False):
        X = num(ch, [c for c in ch.columns if c not in ID])
        X["y"] = ch["Attack"].astype(str).str.strip().values
        parts.append(X)
D = pd.concat(parts, ignore_index=True)
D = cap_per_class(D, 500_000)
D["grp"], D["ts"] = np.int32(0), np.float32(np.nan)
save(D, "nf_ton_v2")

# ---------------- Edge-IIoTset (ML-ready release, one-hot encoded)
parts = []
for f in ["data/raw/edgeiiot/train.csv", "data/raw/edgeiiot/test.csv"]:
    df = pd.read_csv(f, low_memory=False)
    X = num(df, [c for c in df.columns if c not in ("Attack_type", "Attack_label")])
    X.columns = [f"f{i}" for i in range(X.shape[1])]  # raw names embed payload strings
    X["y"] = df["Attack_type"].astype(str).values
    parts.append(X)
D = pd.concat(parts, ignore_index=True)
D["grp"], D["ts"] = np.int32(0), np.float32(np.nan)
save(D, "edgeiiot")

# ---------------- IoT-23 (Zeek flows of 23 real captures)
FEATS = ["l4_proto", "dst_port", "duration", "packets", "packets_rev", "bytes", "bytes_rev", "flow_end_reason",
         "tcp_flags", "tcp_flags_rev", "ppi_len", "ppi_duration", "ppi_roundtrips", "tls_present", "tls_version",
         "tls_cipher_count", "tls_ext_count", "dns_present", "dns_qtype", "dns_rcode", "src_is_local",
         "dst_is_local"]
files = sorted(glob.glob("data/raw/iot23/*.parquet"))
starts = {f: pq.read_table(f, columns=["ts_start"]).column(0).to_numpy().min() for f in files}
files = sorted(files, key=lambda f: starts[f])
parts, nraw = [], 0
for gi, f in enumerate(files):
    df = pq.read_table(f, columns=FEATS + ["label_binary", "label_attack", "ts_start"]).to_pandas()
    nraw += len(df)
    lab = df["label_attack"].where(df["label_attack"].notna() & (df["label_attack"] != ""), df["label_binary"])
    df["y"] = lab.astype(str).str.strip()
    df = cap_per_class(df, 150_000)
    X = num(df, FEATS)
    X["y"], X["grp"], X["ts"] = df["y"].values, np.int32(gi), df["ts_start"].astype("float64").values
    parts.append(X)
D = pd.concat(parts, ignore_index=True)
pd.Series({os.path.basename(f)[:-8]: i for i, f in enumerate(files)}).to_csv("data/proc/iot23_capture_order.csv")
print("iot23 raw flows", nraw, "kept", len(D))
save(D, "iot23")
print("done in %.0fs" % (time.time() - t0))
