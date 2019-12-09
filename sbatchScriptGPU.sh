#!/bin/bash
#SBATCH --job-name=bert2
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=9000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=1-20:00:00

echo $CUDA_VISIBLE_DEVICES
python3 BERT_tuning_and_testing.py "liar"
python3 BERT_tuning_and_testing.py "kaggle"
python3 BERT_tuning_and_testing.py "BS"
