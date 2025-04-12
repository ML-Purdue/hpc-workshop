#!/usr/bin/bash

#SBATCH --account=mlp
#SBATCH --partition=ai
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=56
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00

module load conda
cd /path/to/hpc-workshop
conda activate /path/to/env

OMP_NUM_THREADS=112 \
torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m src.train resnet50_finetuned nocheckpoint
