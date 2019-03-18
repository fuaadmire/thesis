#!/bin/bash
#SBATCH --ntasks=1 --cpus-per-task=10 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=4:00:00

hostname
echo $CUDA_VISIBLE_DEVICES
python3 lstmvis_prep.py
