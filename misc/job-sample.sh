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
conda activate noether

python train.py \
    --image_width 64 \
    --g_dim 128 \
    --z_dim 64 \
    --dataset phys101 \
    --data_root ./data/phys101/phys101/scenarios/ramp \
    --tailor \
    --num_trials 1 \
    --n_past 2 \
    --n_future 20 \
    --num_threads 8 \
    --ckpt_every 10 \
    --inner_crit_mode mse \
    --enc_dec_type vgg \
    --emb_type conserved \
    --num_epochs_per_val 1 \
    --emb_dim 32 \
    --batch_size 4 \
    --num_inner_steps 1 \
    --num_jump_steps 0 \
    --n_epochs 101 \
    --train_set_length 311 \
    --test_set_length 78 \
    --inner_lr .0001 \
    --val_inner_lr .0001 \
    --outer_lr 0.0001 \
    --outer_opt_model_weights \
    --random_weights \
    --only_twenty_degree \
    --frame_step 2 \
    --center_crop 1080 \
    --num_emb_frames 2 \
    --horiz_flip \
    --batch_norm_to_group_norm \
    --reuse_lstm_eps \
    --log_dir ./results/phys101/ramp-$(date +%Y%m%d%H%M%S)/
