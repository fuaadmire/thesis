#!/bin/bash
#SBATCH --job-name=lstm
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=12:00:00

echo $CUDA_VISIBLE_DEVICES
python3 cleaned_lstm_experiments.py "BS"
python3 cleaned_lstm_experiments.py "liar"
python3 cleaned_lstm_experiments.py "kaggle"
python3 cleaned_lstm_experiments.py "FNC"
