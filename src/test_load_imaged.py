#!/usr/bin/env python3
import logging
from pathlib import Path
import sys
from monai.transforms import LoadImaged, EnsureChannelFirstd

# Import the custom image reader and load_image_func from your segmentation module.
from segmentation import CustomImageReader, load_image_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the dataset root; update if necessary.
DATASET_ROOT = Path("C:/Users/niman/OneDrive/Desktop/manifest-1680277513580")
BASE_DIR = DATASET_ROOT / "CT Lymph Nodes"

def has_dicom_files(directory: Path) -> bool:
    """Return True if the directory contains at least one DICOM file."""
    return any(directory.rglob("*.dcm"))

def find_image_and_label(subject_dir: Path):
    """
    Try to find an image and a label for a subject.
    First, look for NIfTI files (.nii or .nii.gz). If none are found, look for directories containing DICOM files.
    For the label, the name should contain "segment" or "mask".
    For the image, ensure the name does NOT contain "segment" or "mask".
    """
    image_path = None
    label_path = None

    # Try NIfTI files first.
    nii_files = list(subject_dir.rglob("*.nii*"))
    if nii_files:
        for file in nii_files:
            name_lower = file.name.lower()
            if "segment" in name_lower or "mask" in name_lower:
                label_path = file
            else:
                image_path = file
        if image_path and label_path:
            return image_path, label_path

    # If no NIfTI files are found, look for directories with DICOM files.
    # For label: directory name contains "segment" or "mask" and has DICOM files.
    for item in subject_dir.rglob("*"):
        if item.is_dir():
            if ("segment" in item.name.lower() or "mask" in item.name.lower()) and has_dicom_files(item):
                label_path = item
                break

    # For image: directory that does NOT contain "segment" or "mask" but has DICOM files.
    for item in subject_dir.rglob("*"):
        if item.is_dir():
            if ("segment" not in item.name.lower() and "mask" not in item.name.lower()) and has_dicom_files(item):
                image_path = item
                break

    return image_path, label_path

def test_load_imaged(image_path: Path, label_path: Path) -> bool:
    try:
        # Create the transforms using the CustomImageReader.
        load_transform = LoadImaged(keys=["image", "label"], reader=CustomImageReader())
        channel_transform = EnsureChannelFirstd(keys=["image", "label"])

        # For DICOM directories, the load_image_func (via CustomImageReader) can accept a directory.
        data = {"image": str(image_path.resolve()), "label": str(label_path.resolve())}
        logger.info(f"Attempting to load data from: {data}")

        # Apply the transforms.
        loaded_data = load_transform(data)
        channeled_data = channel_transform(loaded_data)

        print(f"Image shape: {channeled_data['image'].shape}, dtype: {channeled_data['image'].dtype}")
        print(f"Label shape: {channeled_data['label'].shape}, dtype: {channeled_data['label'].dtype}")
        return True
    except Exception as e:
        logger.error(f"Error during LoadImaged: {e}")
        return False

def main():
    image_path = None
    label_path = None

    # Iterate over subject directories in BASE_DIR that start with "ABD_LYMPH"
    for subject_dir in BASE_DIR.iterdir():
        if not subject_dir.is_dir() or not subject_dir.name.upper().startswith("ABD_LYMPH"):
            continue

        img, lbl = find_image_and_label(subject_dir)
        if img is not None and lbl is not None:
            logger.info(f"Found subject '{subject_dir.name}' with image: {img} and label: {lbl}")
            image_path, label_path = img, lbl
            break

    if image_path is None or label_path is None:
        print("Could not find valid image and label files (or directories) in the dataset.")
        sys.exit(1)

    print(f"Testing with image: {image_path.resolve()}")
    print(f"Testing with label: {label_path.resolve()}")

    if test_load_imaged(image_path, label_path):
        print("LoadImaged test successful!")
    else:
        print("LoadImaged test failed!")

if __name__ == "__main__":
    main()
