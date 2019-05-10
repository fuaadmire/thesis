#!/bin/bash
#SBATCH --job-name=load_relevant
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-20:00:00
python ./load_large_dataset.py
