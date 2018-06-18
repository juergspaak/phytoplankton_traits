#!/bin/bash
#SBATCH -J cont_par
#SBATCH -t 01:05:00
#SBATCH --mem 1000
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python generate_data_continuous_random.py $SLURM_ARRAY_TASK_ID 0.0
