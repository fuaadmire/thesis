#!/bin/bash
#SBATCH --job-name=time_rs
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=9000M
#SBATCH --time=2-00:00:00

echo $CUDA_VISIBLE_DEVICES
python3 random_search-timedistributed.py "liar"
