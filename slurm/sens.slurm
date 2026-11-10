#!/bin/bash
#SBATCH --job-name=fact_sens
#SBATCH --output=logs/sens2_%A_%a.out
#SBATCH --array=0-98%8
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
cd /virtiofs/work/bambang/iot_exp
source ~/venv_iot/bin/activate
export OMP_NUM_THREADS=16
CELLS=(ciciot2023:novel iot23:capture nf_ton_v2:novel)
CONF=("--capacity 0.02:cap0.02" "--capacity 0.05:cap0.05" "--capacity 0.2:cap0.2" "--eps 0.005:eps0.005" "--eps 0.01:eps0.01" "--eps 0.05:eps0.05" "--delay 1:delay1" "--delay 20:delay20" "--alpha 0.1:alpha0.1" "--label_noise 0.05:noise0.05" "--label_noise 0.1:noise0.1")
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((i % 3))]}
cf=${CONF[$(((i / 3) % 11))]}
seed=$((i / 33))
ds=${cell%%:*}; sc=${cell##*:}
args=${cf%%:*}; tag=${cf##*:}
echo "sens $i $ds $sc seed $seed $args"
python code/run.py $ds $sc $seed --threads 16 --methods core $args --tag _$tag
