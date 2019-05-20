#!/bin/bash
#SBATCH --job-name=logregs2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-20:00:00
python ./all_data_logistic_regression.py
