#!/bin/bash
#SBATCH --job-name=annotation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=0-10:00:00
python ./nlp_annotate.py
