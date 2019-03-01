#!/bin/bash
#SBATCH --job-name=MyJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
python3 ./filnavn.py 10000
