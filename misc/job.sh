#!/bin/bash

#SBATCH --job-name=simple-training
#SBATCH --account=eecs545s002w22_class
#SBATCH --partition=gpu
#SBATCH --time=00-10:30:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --export=ALL

cd ~/workspace/eecs545_project

source ~/miniconda3/etc/profile.d/conda.sh
conda activate noether

python main.py \
    --num_epochs 100 \
    --batch_size 4 \
    --outer_lr 1e-3 \
    --inner_lr 1e-3 \
    --num_workers 1 \
    --predict_velocity_diff \
    --num_neighbors 10 \
    --conserve_quantity 'approx' \
    --device 'cuda' \
    --activation 'Sigmoid' \
    --activation_noether 'Sigmoid' \
    --embedding_dim 4
