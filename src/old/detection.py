#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.linalg import eigh
from qtpy import QtWidgets
import napari

###############################################################################
# measure_short_axis
###############################################################################
def measure_short_axis(coords: np.ndarray, affine: np.ndarray) -> (float, np.ndarray):
    """
    Given voxel coords (z, y, x) for one connected component and an affine:
      - Convert coords to world space,
      - Do PCA (smallest eigenvalue => short-axis),
      - Return (short_axis_mm, center_world [x,y,z]).
    """
    if coords.shape[0] < 3:
        mean_vox = np.mean(coords, axis=0)
        mean_vox_reordered = mean_vox[[2,1,0]].reshape(1,3)
        center_world = nib.affines.apply_affine(affine, mean_vox_reordered)[0]
        return 0.0, center_world

    # Reorder coords to (x,y,z)
    coords_reordered = coords[:, [2,1,0]]
    world_coords = nib.affines.apply_affine(affine, coords_reordered)

    mean_pt = np.mean(world_coords, axis=0)
    centered = world_coords - mean_pt
    cov = np.cov(centered, rowvar=False)
    # smallest eigenvalue => minor axis
    eigenvalues, _ = eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    short_axis = 2.0 * np.sqrt(np.min(eigenvalues))

    return short_axis, mean_pt

###############################################################################
# process_patient
###############################################################################
def process_patient(
    patient_folder: str,
    cancer_thresh: float = 15.0,
    predicted_seg: np.ndarray = None,
    predicted_affine: np.ndarray = None
):
    """
    Load the CT volume from `patient_folder`.
    If `predicted_seg` is provided, use that for connected-components classification.
    Otherwise, load ground-truth LN from 'patxxx_labels_LymphNodes.nii.gz'.

    Returns:
      ct_data (np.ndarray)          : The CT volume (Z,Y,X), float32
      classification_map (np.ndarray): 0=bg,1=benign LN,2=cancer LN
      summary (list of dict)         : short-axis stats for each LN
      raw_pred_seg (np.ndarray or None): the raw predicted multi-class seg, if used
    """
    patient_name = os.path.basename(patient_folder)

    # 1) Load CT
    ct_file = os.path.join(patient_folder, f"{patient_name.lower()}_data.nii.gz")
    if not os.path.isfile(ct_file):
        raise FileNotFoundError(f"No CT file found: {ct_file}")

    ct_img = nib.load(ct_file)
    ct_data = ct_img.get_fdata(dtype=np.float32)
    ct_affine = ct_img.affine

    # 2) Decide which segmentation to use
    if predicted_seg is not None:
        seg_data = predicted_seg
        seg_affine = predicted_affine if predicted_affine is not None else ct_affine
        seg_source = "predicted"
    else:
        # Load ground-truth LN
        seg_file = os.path.join(patient_folder, f"{patient_name.lower()}_labels_LymphNodes.nii.gz")
        if not os.path.isfile(seg_file):
            raise FileNotFoundError(f"No ground-truth LN file found: {seg_file}")
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata(dtype=np.float32)
        seg_affine = seg_img.affine
        seg_source = "ground-truth"

    # 3) Connected components on seg_data>0 => short-axis classification
    binary = (seg_data > 0).astype(np.uint8)
    cc, num_components = label(binary)

    classification_map = np.zeros_like(cc, dtype=np.uint8)  # 0=bg,1=benign LN,2=cancer LN
    summary = []

    for comp_id in range(1, num_components+1):
        coords = np.argwhere(cc == comp_id)
        short_axis, center_world = measure_short_axis(coords, seg_affine)
        lbl = 2 if short_axis > cancer_thresh else 1
        classification_map[cc == comp_id] = lbl
        classification_str = "cancerous" if lbl == 2 else "benign"

        summary.append({
            "component": comp_id,
            "short_axis_mm": short_axis,
            "classification": classification_str,
            "center_world": center_world
        })

    raw_pred_seg = seg_data if predicted_seg is not None else None
    print(f"{patient_name}: used {seg_source} seg -> {num_components} LN component(s).")
    return ct_data, classification_map, summary, raw_pred_seg

###############################################################################
# PatientSelectorWidget
###############################################################################
class PatientSelectorWidget(QtWidgets.QWidget):
    """
    A Qt widget to select a patient folder from a combo box and load that patient.
    """
    def __init__(self, patient_folders, load_callback):
        super().__init__()
        self.patient_folders = patient_folders
        self.load_callback = load_callback

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Select Patient:"))

        self.combo = QtWidgets.QComboBox()
        for folder in patient_folders:
            self.combo.addItem(os.path.basename(folder), folder)
        layout.addWidget(self.combo)

        self.load_button = QtWidgets.QPushButton("Load Patient")
        layout.addWidget(self.load_button)
        self.setLayout(layout)

        self.load_button.clicked.connect(self.on_load)

    def on_load(self):
        folder = self.combo.currentData()
        self.load_callback(folder)

###############################################################################
# run_detection_viewer
###############################################################################
def run_detection_viewer(
    dataset_path: str,
    cancer_thresh: float = 15.0,
    predictions: dict = None
):
    """
    Launches a Napari viewer that displays:
      - "CT Volume" in 3D
      - "Classification": short-axis LN classification (0=bg,1=benign,2=cancer)
      - If `predictions` is provided, also shows "Model Prediction" (multi-class).

    `predictions` dict format:
      { "Pat001": (ct_array, seg_pred[, pred_affine]), "Pat002": (...), ... }

    If a patient is missing from predictions (or predictions=None),
    we load ground-truth LN for classification.

    The short-axis threshold is `cancer_thresh`.
    """
    # 1) Gather patient folders
    patient_folders = sorted([
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.startswith("Pat")
    ])
    if not patient_folders:
        print(f"[ERROR] No patient folders found in: {dataset_path}")
        return

    print("Patient folders:", patient_folders)

    # 2) Create Napari viewer (3D mode)
    viewer = napari.Viewer(ndisplay=3)

    # 3) Add empty layers
    ct_layer = viewer.add_image(
        np.zeros((1,1,1), dtype=np.float32),
        name="CT Volume",
        blending="additive",
        rendering="mip"
    )
    classification_layer = viewer.add_labels(
        np.zeros((1,1,1), dtype=np.uint8),
        name="Classification",
        opacity=0.6,
        rendering="translucent"
    )
    classification_layer.color = {
        0: "black",  # background
        1: "green",  # benign LN
        2: "red"     # cancerous LN
    }

    prediction_layer = None
    if predictions is not None:
        # Show raw multi-class segmentation from the model, if it exists
        prediction_layer = viewer.add_labels(
            np.zeros((1,1,1), dtype=np.uint8),
            name="Model Prediction",
            opacity=0.3,
            rendering="translucent"
        )
        # E.g., if your model output is: 0=bg,1=lymph,2=subcar,3=azygos,4=esophagus
        prediction_layer.color = {
            0: "black",
            1: "blue",     # lymph
            2: "yellow",   # subcar
            3: "purple",   # azygos
            4: "cyan"      # esophagus
        }

    # 4) Define callback to load a given patient
    def load_patient(folder):
        pat_id = os.path.basename(folder)

        # If we have predictions and this patient is in the dict, use them
        pred_seg = None
        pred_affine = None
        if predictions is not None and pat_id in predictions:
            pack = predictions[pat_id]
            # Could be (ct_array, seg_pred) or (ct_array, seg_pred, affine)
            if len(pack) == 2:
                ct_array, pred_seg = pack
            elif len(pack) == 3:
                ct_array, pred_seg, pred_affine = pack
            else:
                raise ValueError("predictions dict must have 2 or 3 elements per patient (ct_array, seg_pred, [affine]).")

            ct_data, class_map, summary, raw_pred_seg = process_patient(
                folder,
                cancer_thresh=cancer_thresh,
                predicted_seg=pred_seg,
                predicted_affine=pred_affine
            )
            if prediction_layer is not None and raw_pred_seg is not None:
                prediction_layer.data = raw_pred_seg

            classification_layer.data = class_map
        else:
            # fallback: ground-truth LN
            ct_data, class_map, summary, _ = process_patient(
                folder,
                cancer_thresh=cancer_thresh,
                predicted_seg=None
            )
            classification_layer.data = class_map
            if prediction_layer is not None:
                # Clear or reset the model prediction layer
                prediction_layer.data = np.zeros_like(class_map)

        # Update CT volume
        ct_layer.data = ct_data
        ct_min, ct_max = float(np.min(ct_data)), float(np.max(ct_data))
        ct_layer.contrast_limits = (ct_min, ct_max)

        # Reset the viewer camera
        viewer.reset_view()

        # Print summary of LN short-axis classification
        print(f"\nSummary for {pat_id}:")
        for comp in summary:
            print(
                f"  Component {comp['component']} => short-axis={comp['short_axis_mm']:.1f} mm "
                f"=> {comp['classification']}, center={comp['center_world']}"
            )

    # 5) Create widget & dock it
    selector = PatientSelectorWidget(patient_folders, load_patient)
    viewer.window.add_dock_widget(selector, area='right')

    # 6) Start Napari
    napari.run()