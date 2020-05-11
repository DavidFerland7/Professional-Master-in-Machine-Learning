#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:k20:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6000M

module load python/3.7
source ~/venv/bin/activate
#pip install -r requirements.txt
#wandb on
python -c 'while 1: import ctypes; ctypes.CDLL(None).pause()'
#python ./train.py preds_out_path/preds_output_path_train.out admin_cfg.json --stats_output_path stats_output_path/stats_output_path_train.out -u="eval_user_cfg.json"
