#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:k80:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M

module load python/3.7
module load hdf5
source ~/venv/bin/activate
pip install --no-index tensorflow_gpu
pip install -r requirements.txt
wandb off

python ./generate_azimuth.py preds_out_path/preds_output_path_train.out admin_cfg.json --stats_output_path stats_output_path/stats_output_path_train.out -u="eval_user_cfg.json"
