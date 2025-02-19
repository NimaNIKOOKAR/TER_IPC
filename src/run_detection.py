import streamlit as st
import tempfile
import logging
import nibabel as nib
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.train_detection import detect_diseased_ganglion

# Définition des constantes
MODEL_PATH = "src/models/best_metric_model0.pth"
OUTPUT_DIR = "src/data/results"

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Vérifier si le répertoire de sortie existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Interface Streamlit
st.title("Détection des Ganglions Malades")
st.write("Téléchargez une image médicale au format .nii pour lancer la détection.")
# Configuration de la taille maximale d'upload (500 Mo)

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
        result_image_path = detect_diseased_ganglion(temp_file_path, MODEL_PATH, OUTPUT_DIR)
        logger.info("Détection terminée.")

        st.success("Détection terminée. Voici le résultat :")

        # Afficher le résultat s'il s'agit d'une image
        if result_image_path and os.path.exists(result_image_path):
            result_img = nib.load(result_image_path).get_fdata()
            fig, ax = plt.subplots()
            ax.imshow(result_img[:, :, slice_idx], cmap="jet")  # Affichage avec un autre colormap
            ax.set_title("Résultat de la détection")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.error("Aucun résultat visuel généré.")

