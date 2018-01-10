#!/bin/bash
#SBATCH -J comp_light
#SBATCH -t 01:05:00
#SBATCH --mem 100
#SBATCH -n 1
#SBATCH -N 1

module load Python/3.5.1-foss-2016a
python generate_data_comp_light.py $SLURM_ARRAY_TASK_ID