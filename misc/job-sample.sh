#!/bin/bash

#SBATCH --job-name=simple-training
#SBATCH --account=eecs545s001w22_class
#SBATCH --partition=gpu
#SBATCH --time=00-00:07:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --export=ALL

cd ~/workspace/eecs545_project

source ~/.local/miniconda3/etc/profile.d/conda.sh
conda activate simple 

python main.py \
    --num_epochs 100 \
    --batch_size 4 \
    --outer_lr 1e-4 \
    --inner_lr 1e-4 \
    --num_workers 1 \
    --num_neighbors 10 \
    --conserve_quantity 'approx' \
    --device 'cuda'
