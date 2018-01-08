#!/bin/bash
#
#SBATCH --job-name=datagen
#SBATCH --output=datagen_rand.txt
#
#SBATCH --ntasks=4
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=1000
module load Python/3.5.1-foss-2016a
python data_generation.py
