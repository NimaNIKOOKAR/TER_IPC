import streamlit as st
import tempfile
import torch
import logging
import nibabel as nib
import matplotlib.pyplot as plt
import os

from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    ToTensord,
    EnsureTyped,
    Compose
)
from monai.networks.nets import DenseNet121  # Réseau de classification

# Définition des constantes
MODEL_PATH = "src/models/best_metric_model0.pth"

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    st.error("Le modèle n'a pas été trouvé ! Vérifiez le chemin.")
    st.stop()

# Définir les transformations pour préparer les images
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

# Fonction de détection (classification binaire)
def detect_diseased_ganglion(image_path: str, model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, model_path)

    # Charger et transformer l'image
    data_dict = {"image": image_path}
    transforms = create_transforms()
    data = transforms(data_dict)
    input_tensor = data["image"].unsqueeze(0).to(device)  # Ajouter batch dimension

    # Inférence
    with torch.no_grad():
        output = model(input_tensor)
        prob_malade = torch.softmax(output, dim=1)[0, 1].item()  # Probabilité d'être malade
        prediction = "Malade" if prob_malade > 0.5 else "Sain"

    return prediction, prob_malade, input_tensor.cpu().numpy()[0, 0, :, :, :]

# Interface Streamlit
st.title("Détection des Ganglions Malades")
st.write("Téléchargez une image médicale au format .nii pour lancer la détection.")

uploaded_file = st.file_uploader("Choisir une image NIfTI (.nii)", type=["nii"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.write("### Image téléchargée :")
    st.write(f"{uploaded_file.name}")

    # Charger et afficher une coupe de l'image NIfTI
    nii_img = nib.load(temp_file_path)
    img_data = nii_img.get_fdata()
    slice_idx = img_data.shape[2] // 2  # Sélection du milieu

    fig, ax = plt.subplots()
    ax.imshow(img_data[:, :, slice_idx], cmap="gray")
    ax.set_title("Coupe axiale de l'image")
    ax.axis("off")
    st.pyplot(fig)

    # Lancer la détection
    if st.button("Lancer la détection"):
        logger.info("Démarrage de la détection des ganglions malades...")
        prediction, prob_malade, img_array = detect_diseased_ganglion(temp_file_path, MODEL_PATH)
        logger.info("Détection terminée.")

        # Affichage du résultat
        st.write(f"### Résultat : {prediction}")
        st.write(f"**Probabilité d'être malade : {prob_malade:.2f}**")

        # Afficher l'image annotée avec le résultat
        fig, ax = plt.subplots()
        ax.imshow(img_array[:, :, slice_idx], cmap="gray")
        ax.set_title(f"Résultat : {prediction} (P = {prob_malade:.2f})")
        ax.axis("off")
        st.pyplot(fig)
