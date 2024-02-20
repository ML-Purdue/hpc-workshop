#!/usr/bin/env python3

import torchvision
import mlflow
import torch
import sys

from src.data import Dataset
from src import const


def get_model():
    backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if const.PRETRAINED else None)
    backbone.fc = torch.nn.LazyLinear(len(const.CLASSES))
    return backbone


def fit(model, optimizer, loss, dataloader):
    if const.LOG_REMOTE: mlflow.set_tracking_uri(const.MLFLOW_TRACKING_URI)
    with mlflow.start_run():
        # log hyperparameters
        mlflow.log_params({k: v for k, v in const.__dict__.items() if k == k.upper() and all(s not in k for s in ['DIR', 'PATH'])})
        mlflow.log_param('optimizer_fn', 'Adam')

        interval = max(1, (const.EPOCHS // 10))
        for epoch in range(const.EPOCHS):
            if not (epoch+1) % interval: print('-' * 10)
            epoch_loss = torch.empty(1)

            for batch in dataloader:
                optimizer.zero_grad()

                X, y = batch
                y_pred = model(X)

                batch_loss = loss(y_pred, y)
                batch_loss.backward()
                optimizer.step()

                epoch_loss = torch.vstack([epoch_loss.to(const.DEVICE), batch_loss])
            metrics = {'cross_entropy_loss': epoch_loss[1:].mean().item()}
            mlflow.log_metrics(metrics, step=epoch)
            if not (epoch+1) % interval:
                print(f'epoch\t\t\t: {epoch+1}')
                for key in metrics: print(f'{key}\t\t: {metrics[key]}')
        print('-' * 10)


if __name__ == '__main__':
    const.MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else const.MODEL_NAME

    dataloader = torch.utils.data.DataLoader(Dataset(),
                                             batch_size=const.BATCH_SIZE,
                                             shuffle=True)
    model = get_model().to(const.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=const.LEARNING_RATE)

    fit(model, optimizer, torch.nn.CrossEntropyLoss(), dataloader)
    torch.save(model.state_dict(), const.MODEL_DIR / f'{const.MODEL_NAME}.pt')
