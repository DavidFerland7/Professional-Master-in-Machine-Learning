#!/bin/bash
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=9000M
module load python/3.7
source ~/venv/bin/activate
pip install --no-index tensorflow_gpu
pip install -r requirements.txt
wandb on
python -c 'while 1: import ctypes; ctypes.CDLL(None).pause()'
#python ./models/Transformer_OneHot/train.py
