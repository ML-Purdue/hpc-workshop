#!/usr/bin/env python3

from pathlib import Path
import torch
import os

# directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_CUTOFF = 2 * 60 * 60
MODEL_NAME = 'resnet'
LEARNING_RATE = 1E-3
CHECKPOINTING = True
PRETRAINED = True  # swap this!
BATCH_SIZE = 64
EPOCHS = 5

# ddp
DDP = os.getenv('WORLD_SIZE') is not None
PORT = 1337

# dataset
N_WORKERS = 4
IMAGE_SIZE = (224, 224)
N_CHANNELS = 3
N_CLASSES = 2
IMAGE_SHAPE = (N_CHANNELS,) + IMAGE_SIZE
CLASSES = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

# logging
MLFLOW_TRACKING_URI = 'https://dagshub.com/ML-Purdue/hpc-workshop.mlflow'
LOG_REMOTE = False
