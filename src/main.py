#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging
from time import time
from zipfile import ZipFile
import glob
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# MONAI imports
from monai.data import (
    PersistentDataset,
    Dataset,
    DataLoader,
    pad_list_data_collate
)
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    Spacingd,
    CropForegroundd,
    CenterSpatialCropd,
    SpatialPadd,
    CopyItemsd,
    Lambdad,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    ToTensord,
    EnsureTyped,
    Compose,
    MapTransform,
    Activations,
    AsDiscrete
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Disable torch._dynamo for reproducibility
import torch._dynamo
torch._dynamo.config.disable = True

###############################################################################
# Logging Setup
###############################################################################
def setup_logging(log_file: str = "training_log.txt") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

###############################################################################
# Custom Transform: RemoveKeysd
###############################################################################
class RemoveKeysd(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            d.pop(key, None)
        return d

###############################################################################
# combine_labels: Merge separate label masks into one multi-class label
###############################################################################
def combine_labels(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Combine multiple label arrays into a single multi-class label:
      0: background
      1: lymph
      2: subcar
      3: azygos
      4: esophagus
    """
    lymph = data["label_lymph"]
    subcar = data["label_subcar"]
    azygos = data["label_azygos"]
    esophagus = data["label_esophagus"]

    label = np.zeros_like(lymph, dtype=np.uint8)
    label[lymph > 0] = 1
    label[subcar > 0] = 2
    label[azygos > 0] = 3
    label[esophagus > 0] = 4

    unique, counts = np.unique(label, return_counts=True)
    logger.debug(f"[DEBUG combine_labels] Unique values in label: {dict(zip(unique, counts))}")
    data["label"] = label
    return data

###############################################################################
# Dataset Preparation: Unzip and locate files
###############################################################################
def prepare_data(extract_path: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    benchmark_path = os.path.join(extract_path, "Benchmark")
    zip_path = os.path.join(extract_path, "LyNoS.zip")
    if not os.path.exists(benchmark_path):
        logger.info("Extracting dataset...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info("Dataset unzipped successfully!")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark folder not found at {benchmark_path}")
    logger.info(f"Dataset extracted to: {benchmark_path}")

    image_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_data.nii.gz")))
    lymph_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_LymphNodes.nii.gz")))
    subcar_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_SubCarArt.nii.gz")))
    azygos_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Azygos.nii.gz")))
    esophagus_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Esophagus.nii.gz")))

    if not (image_paths and lymph_label_paths and subcar_label_paths and azygos_label_paths and esophagus_label_paths):
        raise ValueError("One or more image/label files not found. Check dataset structure.")
    logger.info(f"Found {len(image_paths)} CT images and corresponding labels.")
    return image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths

###############################################################################
# Create Transform Pipelines
###############################################################################
def create_transforms(train: bool = True) -> Compose:
    patch_size = (96, 96, 96)  # Training patch size
    
    # Base transforms applied to both training and validation.
    base_transforms = [
        LoadImaged(
            keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"],
            dtype=np.float32
        ),
        EnsureChannelFirstd(
            keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]
        ),
        combine_labels,
        Spacingd(
            keys=["image", "label"],
            pixdim=(2.0, 2.0, 2.0),
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        CopyItemsd(keys=["label"], times=1, names=["label_binary"]),
        Lambdad(
            keys=["label_binary"],
            func=lambda arr: (arr > 0).astype(arr.dtype)
        )
    ]
    
    final_transforms = [
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.int64),
        RemoveKeysd(keys=["label_lymph", "label_subcar", "label_azygos", "label_esophagus", "label_binary"])
    ]
    
    if train:
        
        train_transforms = base_transforms + [
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label_binary",
                spatial_size=patch_size,
                pos=4,       
                neg=1,
                num_samples=4
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=["image", "label"],
                prob=0.5,
                rotate_range=(0.1, 0.1, 0.1),
                translate_range=(10, 10, 10),
                scale_range=(0.1, 0.1, 0.1),
                mode=("bilinear", "nearest")
            ),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        ] + final_transforms
        return Compose(train_transforms)
    else:

        val_transforms = base_transforms + final_transforms
        
        def debug_label(data):
            
            if "label" in data:
                label = data["label"]
                logger.debug(f"[DEBUG val_transform] Post-transform label unique values: {np.unique(label)}; shape: {label.shape}")
            else:
                logger.debug("[DEBUG val_transform] 'label' key not found in data.")
            return data
        
        val_transforms = val_transforms + [debug_label]
        return Compose(val_transforms)

###############################################################################
# Create DataLoaders
###############################################################################
def create_dataloaders(train_data, val_data, batch_size_train=1, batch_size_val=1, num_workers=0):
    cache_dir = "./persistent_cache"
    os.makedirs(cache_dir, exist_ok=True)
    train_transforms = create_transforms(train=True)
    val_transforms = create_transforms(train=False)

    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=cache_dir)
    val_ds = Dataset(data=val_data, transform=val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=num_workers, collate_fn=pad_list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size_val, shuffle=False,
        num_workers=num_workers, collate_fn=pad_list_data_collate
    )
    return train_loader, val_loader

###############################################################################
# Build the UNet Model
###############################################################################
def build_model(device: torch.device) -> torch.nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,  
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=1,
    ).to(device)
    logger.info("Model moved to device successfully!")
    return model

###############################################################################
# Focal Loss for Multi-class Segmentation
###############################################################################
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = -F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

###############################################################################
# Optional: Overfit Single Case for Debugging
###############################################################################
def overfit_single_case(sample_dict, model, optimizer, loss_function, device, num_epochs=20):
    logger.info("Starting overfit test on a single case...")
    transform = create_transforms(train=True)
    ds = Dataset(data=[sample_dict], transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"Overfit epoch {epoch+1}, loss: {loss.item():.4f}")
    logger.info("Overfit test completed.")

###############################################################################
# Training Epoch Function
###############################################################################
def train_epoch(model: torch.nn.Module, loader, loss_function, optimizer, device: torch.device, use_amp: bool = False) -> float:
    model.train()
    epoch_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for batch_idx, batch in enumerate(loader, start=1):
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        epoch_loss += loss.item()
        logger.debug(f"[Train] Batch {batch_idx} => inputs: {inputs.shape}, outputs: {outputs.shape}, loss: {loss.item():.4f}")
        preds = torch.argmax(outputs, dim=1)
        unique_preds = torch.unique(preds)
        logger.debug(f"[DEBUG train_epoch] Batch {batch_idx} predicted unique labels: {unique_preds.cpu().numpy()}")
        if batch_idx == 1:
            unique_vals = torch.unique(labels)
            logger.debug(f"[DEBUG train_epoch] Batch {batch_idx} label unique values: {unique_vals.cpu().numpy()}")
            for c in range(5):
                count_c = torch.sum(labels == c).item()
                logger.debug(f"[DEBUG train_epoch] Batch {batch_idx} class {c} count: {count_c}")

    avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0.0
    logger.info(f"Train epoch loss: {avg_loss:.4f}")
    return avg_loss

###############################################################################
# Validation Epoch Function with Sliding-Window Inference
###############################################################################
def validate_epoch(model: torch.nn.Module, loader, dice_metric: DiceMetric, post_pred, device: torch.device, roi_size=(96,96,96)):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            # Here, we expect full-volume input (post CropForegroundd)
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            logger.debug(f"[Val] Batch {batch_idx}: full volume shape: {val_inputs.shape}")
            # Use sliding-window inference over the full volume.
            outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size=1, predictor=model)
            outputs_post = post_pred(outputs)
            # Convert ground truth to one-hot for Dice metric.
            val_labels_squeezed = torch.squeeze(val_labels, 1)
            val_labels_onehot = F.one_hot(val_labels_squeezed.long(), num_classes=5)
            val_labels_onehot = val_labels_onehot.permute(0, 4, 1, 2, 3).float()
            dice_metric(y_pred=outputs_post, y=val_labels_onehot)
        aggregated = dice_metric.aggregate()
        dice_metric.reset()
        if aggregated is None:
            overall_dice = 0.0
            per_class_dice = torch.zeros(5)
        else:
            overall_dice = torch.nanmean(aggregated).item() if aggregated.ndim > 0 else aggregated.item()
            per_class_dice = aggregated
    logger.info(f"Validation Dice per class: {per_class_dice.cpu().numpy()}")
    logger.info(f"Validation Overall Dice Score: {overall_dice:.4f}")
    return overall_dice, per_class_dice

###############################################################################
# Main Training Function
###############################################################################
def training_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.randn(1, 1, 10, 10, 10, device=device)
            conv_test = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1).to(device)
            _ = conv_test(test_tensor)
            logger.info("GPU test passed!")
        except Exception as e:
            logger.error(f"GPU test failed: {e}")
            device = torch.device("cpu")
            logger.info("Falling back to CPU.")

    extract_path = r"C:\Users\niman\OneDrive\Desktop\TER_IPC\src\cached_lynos"
    (image_paths, lymph_label_paths, subcar_label_paths,
     azygos_label_paths, esophagus_label_paths) = prepare_data(extract_path)

    (train_imgs, val_imgs, train_lymph, val_lymph, train_subcar,
     val_subcar, train_azygos, val_azygos, train_esophagus,
     val_esophagus) = train_test_split(
         image_paths, lymph_label_paths, subcar_label_paths,
         azygos_label_paths, esophagus_label_paths,
         test_size=0.2, random_state=42
     )

    train_data = [
        {
            "image": img,
            "label_lymph": lbl_lymph,
            "label_subcar": lbl_subcar,
            "label_azygos": lbl_azygos,
            "label_esophagus": lbl_esophagus,
        }
        for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus in zip(
            train_imgs, train_lymph, train_subcar, train_azygos, train_esophagus
        )
    ]
    val_data = [
        {
            "image": img,
            "label_lymph": lbl_lymph,
            "label_subcar": lbl_subcar,
            "label_azygos": lbl_azygos,
            "label_esophagus": lbl_esophagus,
        }
        for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus in zip(
            val_imgs, val_lymph, val_subcar, val_azygos, val_esophagus
        )
    ]

    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size_train=1,
        batch_size_val=1,
        num_workers=0
    )

    model = build_model(device)

    # Focal loss with reduced background weight.
    ce_weight = torch.tensor([0.001, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    focal_loss = FocalLoss(gamma=2, weight=ce_weight, reduction="mean")
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

    def combined_loss(outputs, labels):
        # Squeeze labels if needed for focal loss.
        return dice_loss(outputs, labels) + focal_loss(outputs, torch.squeeze(labels, 1))

    loss_function = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    if device.type == "cuda":
        for group in optimizer.param_groups:
            group["capturable"] = True

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean",
        get_not_nans=False
    )

    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=5, dim=1)
    ])

    # Uncomment to run overfit test on a single sample.
    RUN_OVERFIT = False
    if RUN_OVERFIT:
        sample = train_data[0]
        overfit_single_case(sample, model, optimizer, loss_function, device, num_epochs=20)
        return

    # Quick forward-backward test on one batch.
    try:
        model.train()
        batch_data = next(iter(train_loader))
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        logger.info(f"DEBUG: Initial loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        logger.info("DEBUG: Initial forward-backward pass successful.\n")
    except Exception as e:
        logger.error(f"Initial forward-backward pass failed: {e}")
        return

    num_epochs = 80
    best_metric = -1.0
    best_metric_epoch = -1
    use_amp = (device.type == "cuda")

    try:
        for epoch in range(num_epochs):
            start_time = time()
            logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")
            avg_loss = train_epoch(model, train_loader, loss_function, optimizer, device, use_amp)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

            overall_dice, per_class_dice = validate_epoch(model, val_loader, dice_metric, post_pred, device, roi_size=(96,96,96))
            logger.info(f"Validation Overall Dice Score: {overall_dice:.4f}")

            if overall_dice > best_metric:
                best_metric = overall_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                logger.info("Saved new best model!")

            scheduler.step(overall_dice)
            epoch_time = time() - start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Stopping training gracefully.")
    except Exception as e:
        logger.error(f"Global exception during training: {e}")

    logger.info(f"Best validation Overall Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}")

    # Final evaluation: load best model and visualize one full-volume patch.
    try:
        model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    except Exception as e:
        logger.error(f"Error loading best model: {e}")
    model.eval()
    with torch.no_grad():
        # For visualization, run sliding-window inference on the first validation sample.
        for val_batch in val_loader:
            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)
            outputs = sliding_window_inference(val_inputs, (96,96,96), sw_batch_size=1, predictor=model)
            outputs_post = post_pred(outputs)
            val_labels_squeezed = torch.squeeze(val_labels, 1)
            val_labels_onehot = F.one_hot(val_labels_squeezed.long(), num_classes=5)
            val_labels_onehot = val_labels_onehot.permute(0, 4, 1, 2, 3).float()

            plt.figure("Segmentation Results", (12, 6))
            # Choose a mid-slice along the last dimension.
            slice_idx = val_inputs.shape[-1] // 2
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_idx], cmap="gray")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            gt = torch.argmax(val_labels_onehot, dim=1).cpu().numpy()[0]
            plt.imshow(gt[:, slice_idx], cmap="viridis")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            pred = torch.argmax(outputs_post, dim=1).cpu().numpy()[0]
            plt.imshow(pred[:, slice_idx], cmap="viridis")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            break

    logger.info("Training finished.")

if __name__ == "__main__":
    training_main()
