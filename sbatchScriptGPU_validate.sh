#!/bin/bash
#SBATCH --job-name=test_validate
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=30:00:00

echo $CUDA_VISIBLE_DEVICES
python3 validate.py
