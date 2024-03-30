#!/usr/bin/bash

#SBATCH --account=mlp-n
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

module load cuda/12.1.1 cudnn/cuda-12.1_8.9 anaconda
source activate cnn-workshop

cd ~/git/cnn-workshop

MLFLOW_TRACKING_USERNAME=jinensetpal \
MLFLOW_TRACKING_PASSWORD=$MLFLOW_TOKEN \
python -m src.train $MODEL_NAME
