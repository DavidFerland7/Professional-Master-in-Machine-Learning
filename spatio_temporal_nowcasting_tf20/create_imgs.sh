#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M

module load python/3.7
source ~/venv/bin/activate
pip install --no-index tensorflow_gpu
pip install -r requirements.txt
wandb on

python ./create_imgs.py preds_out_path/preds_output_path_train.out admin_cfg.json --region_size $1 --stats_output_path stats_output_path/stats_output_path_train.out -u="eval_user_cfg.json"
