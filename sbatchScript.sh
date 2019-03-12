#!/bin/bash
#SBATCH --job-name=liarlogreg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
python3 ./liar_logistic_regression.py 10000
