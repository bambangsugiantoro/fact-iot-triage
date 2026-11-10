#!/bin/bash
#SBATCH --job-name=factGam
#SBATCH --output=logs/gamma_%A_%a.out
#SBATCH --array=0-17%6
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=6:00:00
cd /virtiofs/work/bambang/iot_exp
source ~/venv_iot/bin/activate
export OMP_NUM_THREADS=16
CELLS=(ciciot2023:novel iot23:capture nf_ton_v2:novel)
GAM=(0.01 0.2)
i=$SLURM_ARRAY_TASK_ID
cell=${CELLS[$((i % 3))]}
g=${GAM[$(((i / 3) % 2))]}
seed=$((i / 6))
ds=${cell%%:*}; sc=${cell##*:}
echo "task $i $ds $sc seed $seed gamma $g on $(hostname)"
python code/run_v3.py $ds $sc $seed --threads 16 --methods factonly --gamma $g --tag _gam${g}v2
