#!/bin/bash
#SBATCH --job-name=prune_sensitivity
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log
#SBATCH --partition=coc-gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32
#SBATCH --time=12:00:00

module load anaconda3/2022.05.0.1
conda activate ml
cd /home/hice1/mortega33/scratch/deep-learning-playground

python train_and_prune.py --all_ratios --prune_method=sensitivity
