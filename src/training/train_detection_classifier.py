import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    EnsureTyped,
    Compose
)
from monai.networks.nets import DenseNet121  # Réseau de classification


# Définir les transformations
def create_transforms():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys="image"),
        ToTensord(keys=["image"]),
        EnsureTyped(keys=["image"], dtype=torch.float),
    ])


# Charger le modèle
def load_model(device: torch.device, model_path: str) -> torch.nn.Module:
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)  # 2 classes : sain/malade
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# Détection des ganglions malades (classification binaire)
def detect_diseased_ganglion(image_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    model = load_model(device, model_path)

    # Préparer les données
    data_dict = {"image": image_path}
    transforms = create_transforms()
    data = transforms(data_dict)
    input_tensor = data["image"].unsqueeze(0).to(device)  # Ajouter batch dimension

    # Inférence
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()  # 0 = sain, 1 = malade

    # Affichage du résultat
    result = "Malade" if prediction == 1 else "Sain"
    print(f"Résultat de la détection : {result}")

    # Visualisation
    plt.imshow(input_tensor.cpu().numpy()[0, 0, :, :, 50], cmap="gray")
    plt.title(f"Résultat : {result}")
    plt.axis("off")
    plt.show()

    return result
