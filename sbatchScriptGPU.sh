#!/bin/bash
#SBATCH --job-name=bert
#SBATCH --ntasks=1 --cpus-per-task=12 --mem=9000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=00:01:00

echo $CUDA_VISIBLE_DEVICES
python3 BERT_tuning_and_testing.py "liar"
