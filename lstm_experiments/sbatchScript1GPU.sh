#!/bin/bash
#SBATCH --job-name=lstm
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=6000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=1-00:00:00

echo $CUDA_VISIBLE_DEVICES
python3 cleaned_lstm_experiments.py "kaggle" 2
python3 cleaned_lstm_experiments.py "BS" 2
python3 cleaned_lstm_experiments.py "liar" 2
python3 cleaned_lstm_experiments.py "kaggle" 16
python3 cleaned_lstm_experiments.py "BS" 16
python3 cleaned_lstm_experiments.py "liar" 16
python3 cleaned_lstm_experiments.py "kaggle" 42
python3 cleaned_lstm_experiments.py "BS" 42
python3 cleaned_lstm_experiments.py "liar" 42
python3 cleaned_lstm_experiments.py "kaggle" 1
python3 cleaned_lstm_experiments.py "BS" 1
python3 cleaned_lstm_experiments.py "liar" 1
python3 cleaned_lstm_experiments.py "kaggle" 4
python3 cleaned_lstm_experiments.py "BS" 4
python3 cleaned_lstm_experiments.py "liar" 4
