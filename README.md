# FACT: Feedback-Adaptive Conformal Triage for IoT intrusion detection

Code, SLURM job scripts and result tables for the manuscript
"Closing the Analyst Loop: Feedback-Adaptive Conformal Triage for IoT Intrusion Detection under Distribution Shift"
(B. Sugiantoro, F. A. Hafizuddin; submitted to The Journal of Supercomputing).

## Layout
- `code/prep.py` – builds the capped parquet files from the public releases (CICIoT2023, NF-ToN-IoT-v2, Edge-IIoTset, IoT-23).
- `code/core.py` – detectors, all conformal baselines, FACT, and the streaming triage simulator (`run_stream`).
  `code/core_v4.py` adds the rare-class guard used by FACT-G (level cap and pooled calibration); `core_v3.py` is the level-cap-only intermediate version.
- `code/run.py` – one (dataset, scenario, seed) cell: `python run.py ciciot2023 novel 0 --tag _v2`.
- `code/aggregate.py`, `code/export.py` – collect the result JSONs into CSVs.
- `code/make_tables.py`, `code/make_figs.py` – produce every table and figure in the paper.
- `slurm/` – job arrays used on the UIN Sunan Kalijaga HPC cluster (one 128-core node; 16 cores per task).
- `results/` – exported CSVs (all runs, summaries, series).

## Environment
Python 3.11, numpy, pandas, scikit-learn, lightgbm, scipy, matplotlib, pyarrow.

## Result tags in `results/all_runs.csv`
- `v2`: main runs (8 cells x 5 seeds, 15 methods); `g2`: FACT-G; `etv2`: ExtraTrees detector;
  `cap*`, `eps*`, `delay*`, `alpha*`, `noise*`, `gam*` ending in `v2`: sensitivity runs (3 cells x 3 seeds).
- Tags without the `v2` suffix (`main`, `g`, `cap0.02`, ...) are preliminary runs of earlier code versions and are not reported in the paper.

## Reproducing the tables and figures
```
python code/make_tables.py results/table_all.csv results/all_runs.csv results/all_info.csv out/
python code/make_figs.py results/series_compact.csv results/all_runs.csv out/ results/all_runs.csv
python code/make_supp.py results/table_all.csv out/supplementary.tex
```

## License
MIT (see LICENSE).
