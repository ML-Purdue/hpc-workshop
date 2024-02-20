#!/usr/bin/env python3

from pathlib import Path
import torch

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'resnet'
LEARNING_RATE = 1E-3
PRETRAINED = True  # swap this!
BATCH_SIZE = 64
EPOCHS = 5

# dataset
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/ML-Purdue/cnn-workshop.mlflow'
LOG_REMOTE = True
