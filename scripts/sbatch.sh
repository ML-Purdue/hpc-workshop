#!/usr/bin/bash

#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --account=csso-h
#SBATCH --gres=gpu:3
#SBATCH --time=2:00:00

module load cuda/12.1.1 cudnn/cuda-12.1_8.9 anaconda
conda activate /scratch/gilbreth/jsetpal/conda/workshop

cd ~/git/hpc-workshop

MLFLOW_TRACKING_USERNAME=$MLFLOW_USERNAME \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
OMP_NUM_THREADS=80 \
torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m src.train resnet50_finetuned nocheckpoint
