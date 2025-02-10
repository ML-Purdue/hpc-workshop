#!/usr/bin/env python3

from torcheval.metrics.toolkit import sync_and_compute
from torcheval.metrics import MulticlassAccuracy
from contextlib import nullcontext
import torch.distributed as dist
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
import torchvision
import numpy as np
import mlflow
import torch
import time
import json
import sys
import os

from src.data import Dataset, DataLoader
from src import const


def get_model():
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if const.PRETRAINED else None)
    backbone.fc = nn.Sequential(nn.Linear(2048, len(const.CLASSES)), nn.Softmax(1))
    return backbone


def fit(model, optimizer, criterion, dataloader, path,
        store=None, init_epoch=0, selected=None, mlflow_run_id=None):
    start_time = time.time()
    selected = selected or {'last': model.state_dict(),
                            'epoch': init_epoch,
                            'acc': 0.0}
    is_primary_rank = not const.DDP or (const.DDP and const.DEVICE == 0)

    with mlflow.start_run(mlflow_run_id) if is_primary_rank else nullcontext():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            metrics = {'cse': [],
                       'acc': MulticlassAccuracy(device=torch.device(const.DEVICE))}

            try:
                for batch in dataloader:
                    optimizer.zero_grad()

                    X, y = [t.to(const.DEVICE) for t in batch]
                    y_pred = model(X)

                    batch_loss = criterion(y_pred, y)
                    batch_loss.backward()
                    optimizer.step()

                    metrics['acc'].update(y_pred.detach(), y.argmax(1))
                    metrics['cse'].append(F.cross_entropy(y_pred, y).item())
            except KeyboardInterrupt:
                break

            metrics = {'cse': np.mean(metrics['cse']),
                       'acc': sync_and_compute(metrics['acc']).item() if const.DDP else metrics['acc'].compute()}
            if const.DDP:
                store.set(f'metric_{const.DEVICE}', json.dumps(metrics))
                dist.barrier(device_ids=[const.DEVICE])

            if not is_primary_rank: continue

            if const.DDP:
                dist_metrics = [json.loads(store.get(f'metric_{rank}')) for rank in range(int(os.environ['WORLD_SIZE']))]
                metrics = {key: np.mean([metric[key] for metric in dist_metrics]) for key in metrics}

            mlflow.log_metrics(metrics, synchronous=False, step=epoch)

            if metrics['acc'] > selected['acc']:
                selected['best'] = deepcopy(model.state_dict())
                selected['epoch'] = epoch
                selected['acc'] = metrics['acc']

            if const.CHECKPOINTING:
                torch.save(optimizer.state_dict(), path / 'optim.pt')
                torch.save(model.state_dict(), path / 'last.pt')

            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')

            if const.TRAIN_CUTOFF is not None and time.time() - start_time >= const.TRAIN_CUTOFF: break

    if is_primary_rank:
        selected['last'] = deepcopy(model.state_dict())

        if 'best' not in selected:
            selected['best'] = deepcopy(model.state_dict())
            selected['epoch'] = epoch
            selected['acc'] = metrics['acc']

        mlflow.log_metrics({'selected_epoch': selected['epoch'],
                            'selected_acc': selected['acc']}, synchronous=False, step=epoch)

    print('-' * 10)
    return epoch, selected


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME
    path = const.MODELS_DIR / const.MODEL_NAME
    (path).mkdir(exist_ok=True, parents=True)

    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    if const.DDP:
        const.DEVICE = int(os.environ['LOCAL_RANK'])
        store = dist.TCPStore('127.0.0.1', const.PORT, is_master=const.DEVICE == 0)
        torch.cuda.set_device(const.DEVICE)
        dist.init_process_group('nccl')
        dist.barrier(device_ids=[const.DEVICE])

    dataloader = DataLoader(Dataset(),
                            batch_size=const.BATCH_SIZE,
                            num_workers=const.N_WORKERS,
                            pin_memory=True,
                            shuffle=True)
    model = get_model().to(const.DEVICE)
    if const.DDP:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[const.DEVICE])
        model.load_state_dict = model.module.load_state_dict
        model.state_dict = model.module.state_dict

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.LEARNING_RATE)

    checkpoint_args = {'init_epoch': 0,
                       'mlflow_run_id': None}
    selected = None
    if const.CHECKPOINTING and len(sys.argv) == 2:  # add extra sys.argv to signify first checkpointing run
        model.load_state_dict(torch.load(path / 'last.pt', map_location=torch.device(const.DEVICE), weights_only=True))
        optimizer.load_state_dict(torch.load(path / 'optim.pt', map_location=torch.device(const.DEVICE), weights_only=True))
        checkpoint_args = json.load(open(path / 'checkpoint_metadata.json'))
        prev_metrics = mlflow.get_run(checkpoint_args['mlflow_run_id']).data.metrics

        selected = {'best': torch.load(path / 'best.pt', map_location='cpu', weights_only=True),
                    'last': model.state_dict(),
                    'epoch': prev_metrics.get('selected_epoch', checkpoint_args['init_epoch']),
                    'acc': prev_metrics.get('selected_acc', prev_metrics.get('acc', 0))}

    completed_epochs, selected = fit(model, optimizer, nn.CrossEntropyLoss(), dataloader, path, store=store if const.DDP else None, selected=selected, **checkpoint_args)

    if const.DDP: dist.destroy_process_group()
    if not const.DDP or (const.DDP and const.DEVICE == 0):
        torch.save(selected['last'], path / 'last.pt')
        torch.save(selected['best'], path / 'best.pt')

        if const.CHECKPOINTING:
            torch.save(optimizer.state_dict(), path / 'optim.pt')
            json.dump({'init_epoch': completed_epochs+1, 'mlflow_run_id': mlflow.last_active_run().info.run_id}, open(path / 'checkpoint_metadata.json', 'w'))
