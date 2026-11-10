#!/bin/bash
#SBATCH --job-name=factG
#SBATCH --output=logs/guard_%A_%a.out
#SBATCH --array=0-39%6
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
cd /virtiofs/work/bambang/iot_exp
source ~/venv_iot/bin/activate
export OMP_NUM_THREADS=16
CELLS=(ciciot2023:novel ciciot2023:capture ciciot2023:prior nf_ton_v2:novel nf_ton_v2:prior edgeiiot:novel edgeiiot:prior iot23:capture)
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((i % 8))]}
seed=$((i / 8))
ds=${cell%%:*}; sc=${cell##*:}
echo "task $i $ds $sc seed $seed on $(hostname)"
python code/run_v3.py $ds $sc $seed --threads 16 --methods guard --tag _g
