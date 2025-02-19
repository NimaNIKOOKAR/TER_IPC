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