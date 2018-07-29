#!/bin/bash
#SBATCH -o output-gpu-%A.%a.out
#SBATCH -p GTX
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --time 48:00:00
#SBATCH --array=0-10

module load cuda/7.5
export OPENMM_PLUGIN_DIR=/home/steboss/sire.app/lib/plugins/

lamvals=(0.000 0.100 0.200 0.300 0.400 0.500 0.600 0.700 0.800 0.900 1.000)
lam=${lamvals[SLURM_ARRAY_TASK_ID]}

echo "lambda is: " $lam

srun  ~/sire.app/bin/python discharging.py  $lam
