#!/bin/bash
#SBATCH --job-name=validate
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=9000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=40:00:00

echo $CUDA_VISIBLE_DEVICES
python3 validate.py
