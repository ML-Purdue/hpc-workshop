#!/usr/bin/bash

#SBATCH --account=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100_32GB
#SBATCH --time=2:00:00

module load cuda cudnn anaconda
source activate cnn-workshop

cd ~/git/cnn-workshop

MLFLOW_TRACKING_USERNAME=jinensetpal \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
python -m src.train $MODEL_NAME
