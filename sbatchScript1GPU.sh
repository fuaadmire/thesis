#!/bin/bash
#SBATCH --job-name=lstm
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=04:00:00

echo $CUDA_VISIBLE_DEVICES
python3 lstm_experiments/cleaned_lstm_experiments.py FNC
