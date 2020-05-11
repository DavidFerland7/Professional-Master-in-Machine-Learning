#!/bin/bash
#SBATCH --time=4:59:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M

module load python/3.7
source ~/venv/bin/activate

#python run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
#python run_exp.py --model=RNN --optimizer=SGD --initial_lr=1.0 --batch_size=20  --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
#python run_exp.py --model=RNN --optimizer=SGD --initial_lr=10.0 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
#python run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best
#python run_exp.py --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.8  --num_epochs=20 --save_best

#python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
#python run_exp.py --model=GRU --optimizer=SGD  --initial_lr=10.0  --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
#python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
#
#python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=256 --num_layers=2 --dp_keep_prob=0.2  --num_epochs=20 --save_best
#python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.5  --num_epochs=20 --save_best
#python run_exp.py --model=GRU --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=4 --dp_keep_prob=0.5  --num_epochs=20 --save_best

python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=512  --num_layers=2 --dp_keep_prob=0.9 --num_epochs=20
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=2048 --num_layers=2 --dp_keep_prob=0.6 --num_epochs=20
python run_exp.py --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.0001 --batch_size=128 --seq_len=35 --hidden_size=1024 --num_layers=6 --dp_keep_prob=0.9 --num_epochs=20
