#!/bin/bash
#SBATCH --job-name=logregs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-20:00:00
python ./last_models_logistic_regression.py
