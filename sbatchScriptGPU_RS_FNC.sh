#!/bin/bash
#SBATCH --job-name=RS_FNC
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=9000M
#SBATCH -p gpu --gres=gpu:titanx:2
#SBATCH --time=4-00:00:00

echo $CUDA_VISIBLE_DEVICES
python3 random_search.py "FNC"
