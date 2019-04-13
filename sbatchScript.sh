#!/bin/bash
#SBATCH --job-name=lstmvisPrep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00
python ./nlp_annotate.py
