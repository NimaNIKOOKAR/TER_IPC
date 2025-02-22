#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

# Import your refactored code from segmentation.py and detection.py
from segmentation import train_segmentation
from detection import run_detection_viewer

def main():
    parser = argparse.ArgumentParser(description="Main script for 3D segmentation & detection with Napari.")
    
    # Required argument: dataset path
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset folder (where 'PatXXX' subfolders exist)."
    )
    
    # Mode of operation: segmentation, detection, or both
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["segmentation", "detection", "both"],
        help="Run only segmentation, only detection, or both."
    )
    
    # Segmentation hyperparameters
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training/validation batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for the segmentation model.")
    parser.add_argument("--checkpoint-path", type=str, default="best_metric_model.pth",
                        help="Path to save or load the best model checkpoint.")
    parser.add_argument("--cache-dir", type=str, default="./persistent_cache",
                        help="Directory to store MONAI's PersistentDataset cache.")
    parser.add_argument("--run-overfit", action="store_true", 
                        help="If set, run a small overfit test on a single sample (debug).")
    
    # Detection parameter: short-axis threshold
    parser.add_argument("--cancer-thresh", type=float, default=15.0,
                        help="Short-axis threshold (mm) above which a lymph node is considered cancerous.")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset_path
    mode = args.mode.lower()

    if not os.path.exists(dataset_path):
        print(f"[ERROR] dataset path not found: {dataset_path}")
        return
    
    # 1) Possibly run segmentation (training)
    if mode in ("segmentation", "both"):
        train_segmentation(
            dataset_path=dataset_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            cache_dir=args.cache_dir,
            checkpoint_path=args.checkpoint_path,
            run_overfit=args.run_overfit
        )
    
    # 2) Possibly run detection (inference + Napari)
    if mode in ("detection", "both"):
        # First, run inference if you want to see the model predictions in Napari
        predictions = run_segmentation_inference(
            dataset_path=dataset_path,
            checkpoint_path=args.checkpoint_path,
            roi_size=(96, 96, 96),
            output_classes=5  # 0=bg,1=lymph,2=subcar,3=azygos,4=esophagus
        )

        # Then open Napari viewer to display classification + (optional) predicted segmentation
        run_detection_viewer(
            dataset_path=dataset_path,
            cancer_thresh=args.cancer_thresh,
            predictions=predictions
        )


if __name__ == "__main__":
    main()
