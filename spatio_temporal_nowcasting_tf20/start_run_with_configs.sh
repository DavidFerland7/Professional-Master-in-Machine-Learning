#!/bin/bash
#SBATCH --time=4:59:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14000M

module load python/3.7
source ~/venv/bin/activate
pip install --no-index tensorflow_gpu
pip install -r requirements.txt
wandb on
python -c 'while 1: import ctypes; ctypes.CDLL(None).pause()'
#python ./train.py preds_out_path/preds_output_path_train.out admin_cfg.json --stats_output_path stats_output_path/stats_output_path_train.out -u="eval_user_cfg.json"
