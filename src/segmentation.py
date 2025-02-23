#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import logging
from time import time
from zipfile import ZipFile
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
    CopyItemsd,
    Lambdad,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ResizeWithPadOrCropd, 
    ToTensord,
    EnsureTyped,
    Compose,
    MapTransform,
    Activations,
    AsDiscrete
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from torch.optim.lr_scheduler import CosineAnnealingLR

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
    logger.debug(f"[DEBUG combine_labels] Unique label values: {dict(zip(unique, counts))}")
    data["label"] = label
    return data

###############################################################################
# prepare_data: Unzip (if needed) and locate image/label files
###############################################################################
def prepare_data(dataset_path: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    benchmark_path = os.path.join(dataset_path, "Benchmark")
    zip_path = os.path.join(dataset_path, "LyNoS.zip")
    if not os.path.exists(benchmark_path) and os.path.isfile(zip_path):
        logger.info("Extracting dataset...")
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)
        logger.info("Dataset unzipped successfully!")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark folder not found at {benchmark_path}")
    logger.info(f"Dataset found at: {benchmark_path}")
    image_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_data.nii.gz")))
    lymph_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_LymphNodes.nii.gz")))
    subcar_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_SubCarArt.nii.gz")))
    azygos_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Azygos.nii.gz")))
    esophagus_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Esophagus.nii.gz")))
    if not (image_paths and lymph_label_paths and subcar_label_paths and azygos_label_paths and esophagus_label_paths):
        raise ValueError("One or more image/label files not found. Check dataset structure or naming conventions.")
    logger.info(f"Found {len(image_paths)} CT images with associated labels.")
    return image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths

###############################################################################
# create_transforms: Training/Validation transforms
###############################################################################
def create_transforms(train: bool = True) -> Compose:
    # Use a larger patch size for fixed-size outputs.
    patch_size = (160, 160, 160)
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
    # Here we enforce a fixed spatial size (using ResizeWithPadOrCropd) before final conversion.
    resize_transform = ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=patch_size)
    final_transforms = [
        resize_transform,
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.int64),
        # Remove extra keys that might vary across samples.
        RemoveKeysd(keys=[
            "label_lymph", "label_subcar", "label_azygos", "label_esophagus",
            "label_binary", "foreground_start_coord", "foreground_end_coord"
        ])
    ]
    if train:
        train_transforms = base_transforms + [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label_binary",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=4,
                allow_smaller=True
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
            RandGaussianNoised(keys="image", prob=0.1, mean=0.0, std=0.1),
            RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.7, 1.5)),
        ] + final_transforms
        return Compose(train_transforms)
    else:
        val_transforms = base_transforms + final_transforms
        def debug_label(data):
            if "label" in data:
                label = data["label"]
                logger.debug(f"[DEBUG val_transform] label unique values: {np.unique(label)}, shape={label.shape}")
            return data
        val_transforms = val_transforms + [debug_label]
        return Compose(val_transforms)

###############################################################################
# create_dataloaders
###############################################################################
def create_dataloaders(
    train_data,
    val_data,
    batch_size_train=2,
    batch_size_val=1,
    num_workers=0,
    cache_dir="./persistent_cache"
):
    os.makedirs(cache_dir, exist_ok=True)
    train_transforms = create_transforms(train=True)
    val_transforms = create_transforms(train=False)
    train_ds = PersistentDataset(data=train_data, transform=train_transforms, cache_dir=cache_dir)
    val_ds = Dataset(data=val_data, transform=val_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_list_data_collate
    )
    return train_loader, val_loader

###############################################################################
# build_model: Construct a deeper 3D UNet
###############################################################################
def build_model(device: torch.device) -> torch.nn.Module:
    # 5-level UNet for increased capacity.
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    logger.info("Model moved to device successfully!")
    return model

###############################################################################
# FocalLoss for Multi-class segmentation
###############################################################################
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="mean"):
        super().__init__()
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
        return loss

###############################################################################
# overfit_single_case: For debugging
###############################################################################
def overfit_single_case(sample_dict, model, optimizer, loss_function, device, num_epochs=20):
    from torch.utils.data import DataLoader
    from monai.data import Dataset
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
# train_epoch: One training epoch
###############################################################################
def train_epoch(
    model: torch.nn.Module,
    loader,
    loss_function,
    optimizer,
    device: torch.device,
    use_amp: bool = False
) -> float:
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
        logger.debug(f"[Train] Batch {batch_idx} => loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(loader) if len(loader) > 0 else 0.0
    logger.info(f"[Train] Epoch average loss: {avg_loss:.4f}")
    return avg_loss

###############################################################################
# validate_epoch: One validation epoch using sliding-window inference
###############################################################################
def validate_epoch(
    model: torch.nn.Module,
    loader,
    dice_metric: DiceMetric,
    post_pred,
    device: torch.device,
    roi_size=(128,128,128)
):
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size=1, predictor=model)
            outputs_post = post_pred(outputs)
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
    logger.info(f"[Val] Dice per class (bg + 4 structures): {per_class_dice.cpu().numpy()}")
    logger.info(f"[Val] Overall Dice Score: {overall_dice:.4f}")
    return overall_dice, per_class_dice

###############################################################################
# train_segmentation: Main training entry
###############################################################################
def train_segmentation(
    dataset_path: str,
    num_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    cache_dir: str = "./persistent_cache",
    checkpoint_path: str = "best_metric_model.pth",
    run_overfit: bool = False
):
    logger.info(f"[INFO] Using dataset_path: {dataset_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
            test_tensor = torch.randn(1, 1, 10, 10, 10, device=device)
            conv_test = torch.nn.Conv3d(1, 2, kernel_size=3, padding=1).to(device)
            _ = conv_test(test_tensor)
            logger.info("[INFO] GPU test passed!")
        except Exception as e:
            logger.error(f"[ERROR] GPU test failed: {e}")
            device = torch.device("cpu")
            logger.info("[INFO] Falling back to CPU.")
    (image_paths, lymph_paths, subcar_paths, azygos_paths, esophagus_paths) = prepare_data(dataset_path)
    (train_imgs, val_imgs,
     train_lymph, val_lymph,
     train_subcar, val_subcar,
     train_azygos, val_azygos,
     train_esophagus, val_esophagus) = train_test_split(
         image_paths, lymph_paths, subcar_paths,
         azygos_paths, esophagus_paths,
         test_size=0.2, random_state=42
     )
    train_data = [
        {
            "image": img,
            "label_lymph": l_lymph,
            "label_subcar": l_subcar,
            "label_azygos": l_azyg,
            "label_esophagus": l_esoph
        }
        for img, l_lymph, l_subcar, l_azyg, l_esoph in zip(
            train_imgs, train_lymph, train_subcar, train_azygos, train_esophagus
        )
    ]
    val_data = [
        {
            "image": img,
            "label_lymph": l_lymph,
            "label_subcar": l_subcar,
            "label_azygos": l_azyg,
            "label_esophagus": l_esoph
        }
        for img, l_lymph, l_subcar, l_azyg, l_esoph in zip(
            val_imgs, val_lymph, val_subcar, val_azygos, val_esophagus
        )
    ]
    train_loader, val_loader = create_dataloaders(
        train_data, val_data,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        num_workers=0,
        cache_dir=cache_dir
    )
    model = build_model(device)
    ce_weight = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    focal_loss = FocalLoss(gamma=2, weight=ce_weight, reduction="mean")
    dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
    def combined_loss(outputs, labels):
        return dice_loss(outputs, labels) + focal_loss(outputs, torch.squeeze(labels, 1))
    loss_function = combined_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Use CosineAnnealingLR for smoother decay.
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    if run_overfit and len(train_data) > 0:
        overfit_single_case(train_data[0], model, optimizer, loss_function, device, num_epochs=20)
    try:
        model.train()
        batch_data = next(iter(train_loader))
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)
        outputs = model(inputs)
        loss_test = loss_function(outputs, labels)
        logger.info(f"[DEBUG] Initial loss: {loss_test.item():.4f}")
        loss_test.backward()
        optimizer.step()
        torch.cuda.synchronize()
        logger.info("[DEBUG] Initial forward-backward pass successful.")
    except Exception as e:
        logger.error(f"Initial forward-backward pass failed: {e}")
        return
    best_metric = -1.0
    best_metric_epoch = -1
    use_amp = (device.type == "cuda")
    try:
        for epoch in range(num_epochs):
            start_time = time()
            logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
            avg_loss = train_epoch(model, train_loader, loss_function, optimizer, device, use_amp)
            dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            post_pred = Compose([
                Activations(softmax=True),
                AsDiscrete(argmax=True, to_onehot=5, dim=1)
            ])
            overall_dice, per_class_dice = validate_epoch(model, val_loader, dice_metric, post_pred, device, roi_size=(160,160,160))
            logger.info(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}, Val Dice: {overall_dice:.4f}")
            if overall_dice > best_metric:
                best_metric = overall_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
                logger.info("[INFO] Saved new best model checkpoint.")
            scheduler.step()
            logger.info(f"Epoch {epoch+1} completed in {time() - start_time:.2f}s.\n")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected. Stopping training gracefully.")
    except Exception as e:
        logger.error(f"Exception during training loop: {e}")
    logger.info(f"[INFO] Best Val Dice: {best_metric:.4f} at epoch {best_metric_epoch}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        logger.error(f"Error loading best model checkpoint: {e}")
    model.eval()
    if len(val_loader) > 0:
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)
                outputs = sliding_window_inference(val_inputs, (160,160,160), sw_batch_size=1, predictor=model)
                outputs_post = post_pred(outputs)
                val_labels_squeezed = torch.squeeze(val_labels, 1)
                val_labels_onehot = F.one_hot(val_labels_squeezed.long(), num_classes=5)
                val_labels_onehot = val_labels_onehot.permute(0, 4, 1, 2, 3).float()
                plt.figure("Segmentation Results", (12, 6))
                slice_idx = val_inputs.shape[-1] // 2
                plt.subplot(1, 3, 1)
                plt.title("CT Volume")
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
    logger.info("[INFO] Training completed.")

###############################################################################
# build_inference_transforms: For inference preprocessing
###############################################################################
def build_inference_transforms():
    return Compose([
        LoadImaged(keys=["image"], dtype=np.float32),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode=("bilinear",)),
        ScaleIntensityd(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

###############################################################################
# run_segmentation_inference: Run model inference on all patient folders
###############################################################################
def run_segmentation_inference(
    dataset_path: str,
    checkpoint_path: str = "best_metric_model.pth",
    roi_size=(128,128,128),
    output_classes=5
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    logger.info(f"[Inference] Loading model from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    post_pred = Compose([
        Activations(softmax=True),
        AsDiscrete(argmax=True, to_onehot=output_classes, dim=1)
    ])
    patient_folders = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.startswith("Pat")
    ])
    logger.info(f"[Inference] Found {len(patient_folders)} patient folders in {dataset_path}")
    inf_transforms = build_inference_transforms()
    predictions = {}
    with torch.no_grad():
        for folder in patient_folders:
            patient_id = os.path.basename(folder).lower()
            ct_file = os.path.join(folder, f"{patient_id}_data.nii.gz")
            if not os.path.isfile(ct_file):
                logger.warning(f"[Inference] Missing CT file for {patient_id}: {ct_file}")
                continue
            data_dict = {"image": ct_file}
            data_dict = inf_transforms(data_dict)
            ct_tensor = data_dict["image"][None].to(device)
            seg_logits = sliding_window_inference(ct_tensor, roi_size, sw_batch_size=1, predictor=model)
            seg_softmax = post_pred(seg_logits)
            seg_label = torch.argmax(seg_softmax, dim=1)
            pred_label = seg_label.squeeze(0).cpu().numpy().astype(np.uint8)
            ct_data = ct_tensor.squeeze(0).squeeze(0).cpu().numpy()
            predictions[patient_id] = (ct_data, pred_label)
            logger.info(f"[Inference] {patient_id}: done, shape={pred_label.shape}")
    return predictions

###############################################################################
# Main entry point
###############################################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="3D Segmentation Training Script")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--mode", type=str, default="segmentation", help="Mode of operation (e.g., segmentation)")
    args = parser.parse_args()
    if args.mode == "segmentation":
        train_segmentation(dataset_path=args.dataset_path, num_epochs=args.epochs, batch_size=args.batch_size)
