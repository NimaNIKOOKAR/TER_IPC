#!/usr/bin/env python3
# debug_overfit.py

import os
import glob
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    Lambdad,
    RandCropByPosNegLabeld,
    RandFlipd,
    ToTensord,
    EnsureTyped
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms.transform import MapTransform

##############################################################################
# 1) Custom transform to remove multiple label files and combine them
##############################################################################
class RemoveKeysd(MapTransform):
    """
    Remove unwanted keys from the data dictionary.
    """
    def __init__(self, keys):
        super().__init__(keys)
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d.pop(key, None)
        return d


def combine_labels(data):
    """
    Combine separate binary label masks into one multi-class label.
    0 = background,
    1 = lymph nodes,
    2 = subclavian carotid arteries,
    3 = azygos,
    4 = esophagus.
    """
    # Assume all are same shape
    label_lymph = data["label_lymph"]
    label_subcar = data["label_subcar"]
    label_azygos = data["label_azygos"]
    label_esophagus = data["label_esophagus"]

    out_label = np.zeros_like(label_lymph, dtype=np.int64)
    out_label[label_lymph > 0]     = 1
    out_label[label_subcar > 0]    = 2
    out_label[label_azygos > 0]    = 3
    out_label[label_esophagus > 0] = 4

    data["label"] = out_label

    # Optional debug
    unique_vals = np.unique(out_label)
    if len(unique_vals) == 1 and unique_vals[0] == 0:
        print("[WARNING] After combining, only background found.")

    return data


##############################################################################
# 2) Create transforms (similar to your main training script),
#    but simpler since we just want to test overfitting on one subject.
##############################################################################
def create_transforms(train=True):
    all_keys = ["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]
    t = [
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        combine_labels,
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # Create a binary label for RandCropByPosNegLabeld
    CopyItemsd(keys=["label"], times=1, names=["label_binary"]),
    Lambdad(
    keys=["label_binary"],
    func=lambda arr: (arr > 0).astype(arr.dtype)
)

    ]

    if train:
        t.extend([
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label_binary",    # used to sample foreground
                spatial_size=(96, 96, 96),
                pos=1, neg=1,
                num_samples=4
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        ])

    t.extend([
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image"], dtype=torch.float),
        EnsureTyped(keys=["label"], dtype=torch.int64),
        RemoveKeysd(keys=["label_lymph", "label_subcar", "label_azygos", "label_esophagus",
                          "label_binary", "foreground_start_coord", "foreground_end_coord"])
    ])
    return Compose(t)


##############################################################################
# 3) Minimal Overfit Function
##############################################################################
def overfit_single_case(sample_dict, device=torch.device("cuda"), epochs=20):
    """
    Attempt to overfit the model on a single case (sample_dict).
    """
    # Wrap the single dict in a list
    single_data = [sample_dict]

    # Create transforms and dataset
    transform = create_transforms(train=True)
    ds = Dataset(data=single_data, transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # Build a small UNet for speed
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,   # 4 classes + background
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=1
    ).to(device)

    # Simple Dice loss
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(loader)

        # Evaluate on the same sample to see if we can overfit
        model.eval()
        with torch.no_grad():
            val_outputs = sliding_window_inference(images, (96, 96, 96), 1, model)
            dice_metric(y_pred=val_outputs.cpu(), y=labels.cpu())
            dice_val = dice_metric.aggregate().item()
            dice_metric.reset()

        model.train()
        print(f"[Overfit] Epoch {epoch+1}/{epochs}, Loss={epoch_loss:.4f}, Dice={dice_val:.4f}")

    print("Overfit test complete. If Dice remains near 0, check data/labels.")


##############################################################################
# 4) Visualization Helpers
##############################################################################
def visualize_raw_nii(image_path, label_path, title_prefix="Raw"):
    """
    Load and display a mid-slice of raw NIfTI files.
    """
    img_nii = nib.load(image_path)
    lbl_nii = nib.load(label_path)

    img_data = img_nii.get_fdata()
    lbl_data = lbl_nii.get_fdata()

    mid_slice = img_data.shape[-1] // 2

    plt.figure(figsize=(10, 4))
    plt.subplot(1,2,1)
    plt.title(f"{title_prefix} Image Slice {mid_slice}")
    plt.imshow(img_data[:, :, mid_slice], cmap="gray")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(f"{title_prefix} Label Slice {mid_slice}")
    plt.imshow(lbl_data[:, :, mid_slice], cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_transformed_sample(sample_dict):
    """
    Apply your training transforms to sample_dict, then visualize
    the middle slice of the resulting torch tensors.
    """
    transform = create_transforms(train=True)
    output = transform(sample_dict)

    image_t = output["image"]  # shape [1, D, H, W] or [1, H, W, D] depending on order
    label_t = output["label"]  # shape [D, H, W] (int64)

    # Convert to numpy
    image_n = image_t.numpy()[0]  # remove channel dimension -> [D, H, W]
    label_n = label_t.numpy()     # [D, H, W]

    # We assume [C, H, W, D] or [C, D, H, W]. If the last dimension is depth,
    # pick mid of that. (Your dimension ordering might differ.)
    depth_dim = image_n.shape[-1]
    mid_slice = depth_dim // 2

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title(f"Transformed Image Slice {mid_slice}")
    plt.imshow(image_n[:, :, mid_slice], cmap="gray")  # if shape [H, W, D]
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(f"Transformed Label Slice {mid_slice}")
    plt.imshow(label_n[:, :, mid_slice], cmap="jet")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


##############################################################################
# 5) Main: Select one subject, visualize raw data, overfit, visualize transform
##############################################################################
if __name__ == "__main__":
    # 1) Paths to your dataset folder
    base_dir = r"C:\Users\niman\OneDrive\Desktop\TER_IPC\src\cached_lynos\Benchmark"
    # Adjust these patterns if needed:
    # example: "Pat01/pat01_data.nii.gz" etc.
    img_paths = sorted(glob.glob(os.path.join(base_dir, "Pat*", "pat*_data.nii.gz")))
    lymph_paths = sorted(glob.glob(os.path.join(base_dir, "Pat*", "pat*_labels_LymphNodes.nii.gz")))
    subcar_paths = sorted(glob.glob(os.path.join(base_dir, "Pat*", "pat*_labels_SubCarArt.nii.gz")))
    azygos_paths = sorted(glob.glob(os.path.join(base_dir, "Pat*", "pat*_labels_Azygos.nii.gz")))
    esophagus_paths = sorted(glob.glob(os.path.join(base_dir, "Pat*", "pat*_labels_Esophagus.nii.gz")))

    # Pick the first subject for debugging (or whichever you like)
    idx = 0
    image_path = img_paths[idx]
    lymph_path = lymph_paths[idx]
    subcar_path = subcar_paths[idx]
    azygos_path = azygos_paths[idx]
    esophagus_path = esophagus_paths[idx]

    print("Using subject paths:")
    print(" image:", image_path)
    print(" lymph:", lymph_path)
    print(" subcar:", subcar_path)
    print(" azygos:", azygos_path)
    print(" esophagus:", esophagus_path)

    # 2) Visualize the raw NIfTI to confirm there's actual label data
    visualize_raw_nii(image_path, lymph_path, title_prefix="Raw (Lymph)")

    # 3) Build a single-sample dict for transforms
    sample_dict = {
        "image": image_path,
        "label_lymph": lymph_path,
        "label_subcar": subcar_path,
        "label_azygos": azygos_path,
        "label_esophagus": esophagus_path,
    }

    # 4) Visualize the transforms on this sample
    visualize_transformed_sample(sample_dict)

    # 5) Overfit on this single sample (or a small set of samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overfit_single_case(sample_dict, device=device, epochs=20)
