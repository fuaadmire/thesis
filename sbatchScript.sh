#!/bin/bash
#SBATCH --job-name=annotation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-12:00:00
python ./liar_logistic_regression.py
