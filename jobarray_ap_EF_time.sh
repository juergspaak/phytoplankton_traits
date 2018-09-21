#!/bin/bash
#SBATCH -J cont_par
#SBATCH -t 00:35:00
#SBATCH --mem 5000
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python sim_ap_EF_time.py $SLURM_ARRAY_TASK_ID
