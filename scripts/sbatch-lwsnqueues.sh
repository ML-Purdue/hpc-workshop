#!/usr/local/bin/bash

#SBATCH --partition=cuda-gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G

#######
# activate your environment in the interactive shell you use to schedule this job
#######

MLFLOW_TRACKING_USERNAME=$MLFLOW_USERNAME \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
OMP_NUM_THREADS=80 \
torchrun --standalone --nnodes=1 --nproc_per_node=gpu -m src.train resnet50_finetuned nocheckpoint
