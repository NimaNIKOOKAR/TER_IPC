# src/training/train_segmentation.py

from datetime import datetime

import torch
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, ScaleIntensityd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, ToTensord, EnsureTyped, Compose, Lambdad
)
from sklearn.model_selection import train_test_split

from src.training.train_utils import train_epoch, validate_epoch
from src.utils.data_preparation import prepare_data
from src.utils.logger import setup_logging_training
from src.utils.transforms import combine_labels, RemoveKeysd

logger = setup_logging_training("logs/training_log.txt")

def create_transforms_training(train: bool = True) -> Compose:
    """
    Crée un pipeline de transformations pour l'entraînement ou la validation.
    """
    transforms = [
        LoadImaged(keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]),
        EnsureChannelFirstd(keys=["image", "label_lymph", "label_subcar", "label_azygos", "label_esophagus"]),
        Lambdad(keys=["label_lymph", "label_subcar", "label_azygos", "label_esophagus"], func=lambda x: x),
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


def build_model(device: torch.device) -> torch.nn.Module:
    """
    Construit et retourne un modèle UNet 3D.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    logger.info("Modèle chargé avec succès sur le device.")
    return model


def training_main(num_epochs: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation du device: {device}")

    # Préparation des données
    extract_path = r"data/cached_lynos/"
    image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths = prepare_data(extract_path)

    # Division des données
    (train_imgs, val_imgs, train_lymph, val_lymph, train_subcar, val_subcar,
     train_azygos, val_azygos, train_esophagus, val_esophagus) = train_test_split(
        image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths,
        test_size=0.2, random_state=42
    )

    # Création des datasets et dataloaders
    train_data = [{"image": img, "label_lymph": lbl_lymph, "label_subcar": lbl_subcar,
                   "label_azygos": lbl_azygos, "label_esophagus": lbl_esophagus}
                  for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus
                  in zip(train_imgs, train_lymph, train_subcar, train_azygos, train_esophagus)]
    val_data = [{"image": img, "label_lymph": lbl_lymph, "label_subcar": lbl_subcar,
                 "label_azygos": lbl_azygos, "label_esophagus": lbl_esophagus}
                for img, lbl_lymph, lbl_subcar, lbl_azygos, lbl_esophagus
                in zip(val_imgs, val_lymph, val_subcar, val_azygos, val_esophagus)]

    train_loader = DataLoader(Dataset(data=train_data, transform=create_transforms_training(train=True)),
                             batch_size=4, shuffle=True, num_workers=0, collate_fn=pad_list_data_collate)
    val_loader = DataLoader(Dataset(data=val_data, transform=create_transforms_training(train=False)),
                           batch_size=1, shuffle=False, num_workers=0, collate_fn=pad_list_data_collate)

    # Construction du modèle, de la fonction de perte et de l'optimiseur
    model = build_model(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Boucle d'entraînement
    # num_epochs = 1
    best_metric = -1.0
    best_metric_epoch = -1
    use_amp = device.type == "cuda"

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(model, train_loader, loss_function, optimizer, device, use_amp)
        logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        dice_score = validate_epoch(model, val_loader, dice_metric, device)
        logger.info(f"Validation Dice Score: {dice_score:.4f}")

        if dice_score > best_metric:
            best_metric = dice_score
            best_metric_epoch = epoch + 1
            torch.save(model.state_dict(), f"models/best_metric_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            logger.info("Nouveau meilleur modèle sauvegardé!")

    logger.info(f"Meilleur score Dice: {best_metric:.4f} à l'epoch {best_metric_epoch}")

