# FACT: Feedback-Adaptive Conformal Triage for IoT intrusion detection

Code, SLURM job scripts and result tables for the manuscript
"Closing the Analyst Loop: Feedback-Adaptive Conformal Triage for IoT Intrusion Detection under Distribution Shift"
(B. Sugiantoro, F. A. Hafizuddin; submitted to The Journal of Supercomputing).

## Layout
- `code/prep.py` – builds the capped parquet files from the public releases (CICIoT2023, NF-ToN-IoT-v2, Edge-IIoTset, IoT-23).
- `code/core.py` – detectors, all conformal baselines, FACT, and the streaming triage simulator (`run_stream`).
  `code/core_v3.py` is the same file plus the rare-class guard (FACT-G).
- `code/run.py` – one (dataset, scenario, seed) cell: `python run.py ciciot2023 novel 0 --tag _v2`.
- `code/aggregate.py`, `code/export.py` – collect the result JSONs into CSVs.
- `code/make_tables.py`, `code/make_figs.py` – produce every table and figure in the paper.
- `slurm/` – job arrays used on the UIN Sunan Kalijaga HPC cluster (one 128-core node; 16 cores per task).
- `results/` – exported CSVs (all runs, summaries, series).

## Environment
Python 3.11, numpy, pandas, scikit-learn, lightgbm, scipy, matplotlib, pyarrow.
