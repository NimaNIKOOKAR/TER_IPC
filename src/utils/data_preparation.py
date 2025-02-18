# src/utils/data_preparation.py

import os
import re
import glob
import nibabel as nib
import requests
import tempfile
from zipfile import ZipFile
from urllib.parse import urlparse
from typing import List, Tuple


def load_nii_from_zip(zip_url: str, file_path: str) -> nib.Nifti1Image:
    """
    Télécharge un fichier zip (si nécessaire) et extrait un fichier NIfTI.
    """
    parsed_url = urlparse(zip_url)
    clean_filename = os.path.basename(parsed_url.path.split("?")[0])
    clean_filename = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_filename)
    zip_filename = os.path.join("./cached_lynos", clean_filename)

    if not os.path.exists(zip_filename):
        response = requests.get(zip_url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(zip_filename), exist_ok=True)
        with open(zip_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    with ZipFile(zip_filename, "r") as archive:
        with archive.open(file_path) as nii_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as temp_nii:
                temp_nii.write(nii_file.read())
                temp_nii_path = temp_nii.name

    img = nib.load(temp_nii_path)
    img_data = img.get_fdata()
    img_in_memory = nib.Nifti1Image(img_data, img.affine, img.header)
    os.remove(temp_nii_path)
    return img_in_memory


def prepare_data(extract_path: str) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Prépare les données en extrayant les chemins des images et des labels.
    """
    benchmark_path = os.path.join(extract_path, "Benchmark")
    zip_path = os.path.join(extract_path, "LyNoS.zip")

    if not os.path.exists(benchmark_path):
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

    image_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_data.nii.gz")))
    lymph_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_LymphNodes.nii.gz")))
    subcar_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_SubCarArt.nii.gz")))
    azygos_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Azygos.nii.gz")))
    esophagus_label_paths = sorted(glob.glob(os.path.join(benchmark_path, "Pat*", "pat*_labels_Esophagus.nii.gz")))

    return image_paths, lymph_label_paths, subcar_label_paths, azygos_label_paths, esophagus_label_paths