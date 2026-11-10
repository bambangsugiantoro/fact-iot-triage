#!/bin/bash
#SBATCH --job-name=fact_det
#SBATCH --output=logs/det_%A_%a.out
#SBATCH --array=0-8%8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
cd /virtiofs/work/bambang/iot_exp
source ~/venv_iot/bin/activate
export OMP_NUM_THREADS=16
CELLS=(nf_ton_v2:novel iot23:capture edgeiiot:novel)
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((i % 3))]}
seed=$((i / 3))
ds=${cell%%:*}; sc=${cell##*:}
echo "det $i $ds $sc seed $seed"
python code/run.py $ds $sc $seed --threads 16 --methods core --detector et --tag _et
