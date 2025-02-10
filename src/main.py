#!/usr/bin/env python
# coding: utf-8

import os
# Set the default timeout for fsspec via an environment variable.
os.environ["FSSPEC_DEFAULT_TIMEOUT"] = "3600"

import re
import logging
import tempfile
from time import time
from zipfile import ZipFile
from urllib.parse import urlparse
import glob
from typing import Any, Dict, List, Tuple

import torch
import monai
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split

from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    ToTensord,
    EnsureTyped,
    Compose,
    Lambdad,
    MapTransform
)
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference

import torch._dynamo
torch._dynamo.config.disable = True

# Import datasets and DownloadConfig from Hugging Face
from datasets import load_dataset, DownloadConfig

###############################################################################
# Logging Setup: Log to both console and file.
###############################################################################
def setup_logging(log_file: str = "training_log.txt") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

###############################################################################
# Custom Transform to Remove Unwanted Keys
###############################################################################
class RemoveKeysd(MapTransform):
    """
    Remove unwanted keys from the data dictionary.
    """
    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for key in self.keys:
            d.pop(key, None)
        return d

###############################################################################
# Utility Functions
###############################################################################
def load_nii_from_zip(zip_url: str, file_path: str) -> nib.Nifti1Image:
    """
    Download a zip file (if needed) and extract a NIfTI file.
    """
    parsed_url = urlparse(zip_url)
    clean_filename = os.path.basename(parsed_url.path.split("?")[0])
    clean_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_filename)
    zip_filename = os.path.join("./cached_lynos", clean_filename)
    
    if not os.path.exists(zip_filename):
        logger.info(f"Downloading {zip_url} ...")
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()
            os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
            with open(zip_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            logger.error(f"Error downloading {zip_url}: {e}")
            raise

    try:
        with ZipFile(zip_filename, "r") as archive:
            with archive.open(file_path) as nii_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_nii:
                    temp_nii.write(nii_file.read())
                    temp_nii_path = temp_nii.name
    except Exception as e:
        logger.error(f"Error extracting {file_path} from {zip_filename}: {e}")
        raise

    try:
        img = nib.load(temp_nii_path)
        img_data = img.get_fdata()
        img_in_memory = nib.Nifti1Image(img_data, img.affine, img.header)
    finally:
        os.remove(temp_nii_path)
    
    return img_in_memory

def combine_labels(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Combine separate label masks into one multi-class label.
    
    Order of assignment:
      - Background: 0
      - Lymph nodes: 1 (where label_lymph > 0)
      - Subcarotid: 2 (where label_subcar > 0)
      - Azygos: 3 (where label_azygos > 0)
      - Esophagus: 4 (where label_esophagus > 0)
    """
    label = np.zeros_like(data["label_lymph"], dtype=np.int64)
    label[data["label_lymph"] > 0] = 1
    label[data["label_subcar"] > 0] = 2
    label[data["label_azygos"] > 0] = 3
    label[data["label_esophagus"] > 0] = 4
    data["label"] = label
    return data

###############################################################################
# Data Preparation using the Hugging Face datasets library
###############################################################################
def prepare_data_from_datasets() -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Load the LyNoS dataset using the Hugging Face datasets library.
    
    Assumes the dataset at "andreped/LyNoS" has the keys:
      - "data": CT image file path,
      - "labels_LymphNodes": lymph nodes mask path,
      - "labels_SubCarArt": subcarotid arteries mask path,
      - "labels_Azygos": azygos mask path,
      - "labels_Esophagus": esophagus mask path.
    
    Adjust the keys if needed.
    """
    download_config = DownloadConfig(max_retries=10)
    ds = load_dataset("andreped/LyNoS", split="train", download_config=download_config)
    logger.info(f"Loaded dataset with {ds.num_rows} CT scans from Hugging Face datasets.")
    
    image_paths           = ds["data"]
    lymph_label_paths     = ds["labels_LymphNodes"]
    subcar_label_paths    = ds["labels_SubCarArt"]
    azygos_label_paths    = ds["labels_Azygos"]
    esophagus_label_paths = ds["labels_Esophagus"]
    
    logger.info("Extracted file paths for CT images and label masks.")
    return image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths

# Top-level identity function (needed for pickling)
def identity(x):
    return x

def create_transforms(train: bool = True) -> Compose:
    """
    Create a MONAI transform pipeline for training or validation.
    """
    transforms = [
        LoadImaged(keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]),
        EnsureChannelFirstd(keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]),
        Lambdad(keys=["label_lymph", "label_subcar", "label_azygos", "label_esophagus"], func=identity),
        combine_labels,
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    if train:
        transforms.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                num_samples=4
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        ])
    transforms.extend([
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["label"], dtype=torch.int64),
        RemoveKeysd(keys=["label_lymph", "label_subcar", "label_azygos", "label_esophagus",
                           "foreground_start_coord", "foreground_end_coord"])
    ])
    return Compose(transforms)

def create_dataloaders(train_data: List[Dict[str, str]],
                       val_data: List[Dict[str, str]],
                       batch_size_train: int = 4,
                       batch_size_val: int = 1,
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.
    """
    train_ds = Dataset(data=train_data, transform=create_transforms(train=True))
    val_ds   = Dataset(data=val_data, transform=create_transforms(train=False))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, collate_fn=pad_list_data_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False,
                            num_workers=num_workers, collate_fn=pad_list_data_collate, pin_memory=True)
    return train_loader, val_loader

def build_model(device: torch.device) -> torch.nn.Module:
    """
    Build and return a 3D UNet model with 5 output channels.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    logger.info("Model moved to device successfully!")
    return model

###############################################################################
# Training and Validation Functions
###############################################################################
def train_epoch(model: torch.nn.Module, 
                loader: DataLoader, 
                loss_function: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                use_amp: bool = False) -> float:
    """
    Run one epoch of training.
    """
    model.train()
    epoch_loss = 0.0
    scaler = torch.amp.GradScaler() if use_amp else None
    
    for batch in loader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast():
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

def validate_epoch(model: torch.nn.Module, 
                   loader: DataLoader,
                   dice_metric: DiceMetric,
                   device: torch.device) -> float:
    """
    Run one epoch of validation and compute the Dice metric.
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

###############################################################################
# Main Training Function
###############################################################################
def training_main():
    # Device Setup
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
    
    # Data Preparation using Hugging Face datasets.
    image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths = prepare_data_from_datasets()
    
    # Split the data into training and validation sets (80/20 split).
    (train_imgs, val_imgs, 
     train_lymph, val_lymph, 
     train_subcar, val_subcar, 
     train_azygos, val_azygos, 
     train_esophagus, val_esophagus) = train_test_split(
         image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths,
         test_size=0.2, random_state=42
    )
    
    # Build training/validation dictionaries.
    train_data = [
        {
            "image": img,
            "label_lymph": lbl_lymph,
            "label_subcar": lbl_subcar,
            "label_azygos": lbl_azygos,
            "label_esophagus": lbl_esophagus,
        }
        for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus 
        in zip(train_imgs, train_lymph, train_subcar, train_azygos, train_esophagus)
    ]
    val_data = [
        {
            "image": img,
            "label_lymph": lbl_lymph,
            "label_subcar": lbl_subcar,
            "label_azygos": lbl_azygos,
            "label_esophagus": lbl_esophagus,
        }
        for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus 
        in zip(val_imgs, val_lymph, val_subcar, val_azygos, val_esophagus)
    ]
    
    train_loader, val_loader = create_dataloaders(train_data, val_data, num_workers=0)
    
    # Build Model, Loss, Optimizer, and Metrics.
    model = build_model(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for group in optimizer.param_groups:
        group["capturable"] = True
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    
    # Debug: One forward-backward pass on one batch.
    logger.info("DEBUG: Testing one forward-backward pass on one batch")
    model.train()
    batch_data = next(iter(train_loader))
    inputs = batch_data["image"].to(device)
    labels = batch_data["label"].to(device)
    logger.info(f"Unique labels in batch: {torch.unique(labels)}; shape: {labels.shape}")
    outputs = model(inputs)
    logger.info(f"Model outputs shape: {outputs.shape}")
    loss = loss_function(outputs, labels)
    logger.info(f"Loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    logger.info("DEBUG: Forward-backward pass successful.\n")
    
    # Training Loop.
    num_epochs = 50
    best_metric = -1.0
    best_metric_epoch = -1
    use_amp = device.type == "cuda"
    
    for epoch in range(num_epochs):
        start_time = time()
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, train_loader, loss_function, optimizer, device, use_amp)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
        dice_score = validate_epoch(model, val_loader, dice_metric, device)
        logger.info(f"Validation Dice Score: {dice_score:.4f}")
    
        if dice_score > best_metric:
            best_metric = dice_score
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), "best_metric_model.pth")
            logger.info("Saved new best model!")
    
        epoch_time = time() - start_time
        logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds.\n")
    
    logger.info(f"Best validation Dice Score: {best_metric:.4f} at epoch {best_metric_epoch}")
    
    # Visualization of Results using the best model.
    model.load_state_dict(torch.load("best_metric_model.pth", map_location=device))
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch["image"].to(device)
            val_labels = val_batch["label"].to(device)
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
    
            plt.figure("Segmentation Results", (12, 6))
            plt.subplot(1, 3, 1)
            plt.title("Input Image")
            plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, 50], cmap="gray")
            plt.axis("off")
    
            plt.subplot(1, 3, 2)
            plt.title("Ground Truth")
            plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, 50])
            plt.axis("off")
    
            plt.subplot(1, 3, 3)
            plt.title("Prediction")
            plt.imshow(torch.argmax(val_outputs, dim=1).cpu().numpy()[0, :, :, 50])
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    training_main()
