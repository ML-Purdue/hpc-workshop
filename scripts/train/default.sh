#!/usr/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --account=csso-e
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

module load cuda/12.1.1 cudnn/cuda-12.1_8.9 anaconda
conda activate /scratch/gilbreth/jsetpal/conda/workshop

cd /path/to/hpc-workshop

OMP_NUM_THREADS=80 \
torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m src.train resnet50_finetuned nocheckpoint
