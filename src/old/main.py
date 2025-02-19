#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# Import your refactored code from segmentation.py and detection.py
from segmentation import train_segmentation, run_segmentation_inference
from detection import run_detection_viewer

# Constants (replace these with your desired values)
DATASET_PATH = "\\cached_lynos\\Benchmark"
MODE = "both"  # Choose from "segmentation", "detection", or "both"

# Segmentation hyperparameters
EPOCHS = 2
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "best_metric_model.pth"
CACHE_DIR = "./persistent_cache"
RUN_OVERFIT = False  # Set to True for a small overfit test

# Detection parameter: short-axis threshold
CANCER_THRESH = 15.0

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] dataset path not found: {DATASET_PATH}")
        return

    # 1) Possibly run segmentation (training)
    if MODE in ("segmentation", "both"):
        train_segmentation(
            dataset_path=DATASET_PATH,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            cache_dir=CACHE_DIR,
            checkpoint_path=CHECKPOINT_PATH,
            run_overfit=RUN_OVERFIT
        )

    # 2) Possibly run detection (inference + Napari)
    if MODE in ("detection", "both"):
        # First, run inference if you want to see the model predictions in Napari
        predictions = run_segmentation_inference(
            dataset_path=DATASET_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            roi_size=(96, 96, 96),
            output_classes=5  # 0=bg,1=lymph,2=subcar,3=azygos,4=esophagus
        )

        # Then open Napari viewer to display classification + (optional) predicted segmentation
        run_detection_viewer(
            dataset_path=DATASET_PATH,
            cancer_thresh=CANCER_THRESH,
            predictions=predictions
        )

if __name__ == "__main__":
    main()
