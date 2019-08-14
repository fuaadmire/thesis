#!/bin/bash
#SBATCH --job-name=RS_FNC
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=3-20:00:00

echo $CUDA_VISIBLE_DEVICES
python3 random_search.py "FNC"
