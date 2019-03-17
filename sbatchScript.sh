#!/bin/bash
#SBATCH --job-name=lstmvisPrep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
python3 ./lstmvis_prep.py
