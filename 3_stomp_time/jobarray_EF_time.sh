#!/bin/bash
#SBATCH -J cont_par
#SBATCH -t 01:05:00
#SBATCH --mem 500
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python coexistence_EF.py $SLURM_ARRAY_TASK_ID
