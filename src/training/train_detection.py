import os
import logging
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.data import Dataset, DataLoader
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    ToTensord,
    EnsureTyped,
    Compose,
    Lambdad,
    MapTransform
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

from src.utils.logger import setup_logging_detection

logger = setup_logging_detection()


# Définir les transformations pour les données d'entrée
def create_transforms_detection() -> Compose:
    transforms = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys="image"),
        CropForegroundd(keys=["image"], source_key="image"),
        ToTensord(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float),
    ]
    return Compose(transforms)


# Charger le modèle entraîné
def load_model(device: torch.device, model_path: str) -> torch.nn.Module:
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Fonction principale pour détecter les ganglions malades
def detect_diseased_ganglion(image_path: str, model_path: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Charger le modèle
    model = load_model(device, model_path)

    # Préparer les données d'entrée
    data_dict = {"image": image_path}
    transforms = create_transforms_detection()
    data = transforms(data_dict)
    input_tensor = data["image"].unsqueeze(0).to(device)  # Ajouter une dimension de batch

    # Effectuer l'inférence
    with torch.no_grad():
        output = sliding_window_inference(input_tensor, (96, 96, 96), 4, model)

    # Post-traitement des résultats
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Convertir en segmentation finale

    # Identifier les ganglions malades (classe 1)
    diseased_ganglia = (output == 1).astype(np.uint8)

    # Sauvegarder les résultats
    output_image = nib.Nifti1Image(diseased_ganglia, np.eye(4))
    output_path = os.path.join(output_dir, "diseased_ganglia.nii.gz")
    nib.save(output_image, output_path)
    logger.info(f"Saved diseased ganglia detection results to: {output_path}")

    # Afficher les résultats
    plt.figure("Diseased Ganglia Detection", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(input_tensor.cpu().numpy()[0, 0, :, :, 50], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Diseased Ganglia")
    plt.imshow(diseased_ganglia[:, :, 50], cmap="jet")
    plt.axis("off")
    plt.show()


