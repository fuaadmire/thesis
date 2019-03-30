#!/bin/bash
#SBATCH --job-name=lstmvisPrep
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:4
#SBATCH --time=12:00:00

echo $CUDA_VISIBLE_DEVICES
python3 lstmvis_prep.py
