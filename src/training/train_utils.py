# src/training/train_utils.py

import torch
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference


def train_epoch(model: torch.nn.Module, loader, loss_function, optimizer, device, use_amp=False):
    """
    Exécute une époque d'entraînement.
    """
    model.train()
    epoch_loss = 0.0
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validate_epoch(model: torch.nn.Module, loader, dice_metric, device):
    """
    Exécute une époque de validation.
    """
    model.eval()
    with torch.no_grad():
        for batch in loader:
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 1, model)
            dice_metric(y_pred=val_outputs.cpu(), y=val_labels.cpu())
            torch.cuda.empty_cache()
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    return dice_score