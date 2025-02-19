import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label
from scipy.linalg import eigh
import napari
from qtpy import QtWidgets

# -------------------------------
# Functions for Data Loading & Measurement
# -------------------------------
def load_nifti(nifti_path):
    """
    Loads a NIfTI file using nibabel.
    Returns:
      data: a NumPy array with shape (Z, Y, X) (float32)
      affine: a 4x4 matrix mapping voxel indices to world coordinates.
    """
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    return data, affine

def measure_short_axis(coords, affine):
    """
    Given voxel coordinates (in [z, y, x] order) and an affine,
    convert them to world coordinates and compute the short-axis via PCA.
    
    Returns:
      short_axis_mm: estimated short-axis diameter in mm,
      center_world: mean point in world coordinates (x, y, z)
    """
    if coords.shape[0] < 3:
        return 0.0, np.mean(coords, axis=0)
    # Reorder from (z, y, x) to (x, y, z) for nibabel.apply_affine.
    coords_reordered = coords[:, [2, 1, 0]]
    world_coords = nib.affines.apply_affine(affine, coords_reordered)
    mean_pt = np.mean(world_coords, axis=0)
    centered = world_coords - mean_pt
    cov = np.cov(centered, rowvar=False)
    eigenvalues, _ = eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    short_axis = 2.0 * np.sqrt(np.min(eigenvalues))
    return short_axis, mean_pt

def process_patient(patient_folder, cancer_thresh=15.0):
    """
    Process a patient folder:
      - Loads CT and lymph node segmentation.
      - Computes connected components on the segmentation.
      - For each lymph node, measures the 3D short-axis and classifies it:
          label 1: benign (short-axis <= cancer_thresh)
          label 2: cancerous (short-axis > cancer_thresh)
    
    Returns:
      ct_data, ct_affine, new_seg (multi-label segmentation), summary (list of dicts)
    """
    patient_name = os.path.basename(patient_folder)
    ct_file = os.path.join(patient_folder, f"{patient_name.lower()}_data.nii.gz")
    seg_file = os.path.join(patient_folder, f"{patient_name.lower()}_labels_LymphNodes.nii.gz")
    
    ct_data, ct_affine = load_nifti(ct_file)
    seg_data, seg_affine = load_nifti(seg_file)
    
    if ct_data.shape != seg_data.shape:
        print(f"WARNING: CT and segmentation shapes do not match for {patient_name}!")
    
    # Create a binary mask (nonzero = lymph node).
    binary = (seg_data > 0).astype(np.uint8)
    cc, num_components = label(binary)
    print(f"Found {num_components} connected components in {patient_name}.")
    
    # Create a new multi-label segmentation: 0=background, 1=benign, 2=cancerous.
    new_seg = np.zeros_like(cc, dtype=np.uint8)
    summary = []
    for comp in range(1, num_components + 1):
        coords = np.argwhere(cc == comp)  # [z, y, x]
        if coords.shape[0] < 3:
            continue
        short_axis, center_world = measure_short_axis(coords, seg_affine)
        label_val = 2 if short_axis > cancer_thresh else 1
        new_seg[cc == comp] = label_val
        summary.append({
            'component': comp,
            'short_axis_mm': short_axis,
            'classification': 'cancerous' if label_val == 2 else 'benign',
            'center_world': center_world
        })
        print(f"  Component {comp}: short-axis = {short_axis:.1f} mm -> {('cancerous' if label_val==2 else 'benign')}")
    
    return ct_data, ct_affine, new_seg, summary

# -------------------------------
# Qt Widget for Patient Selection
# -------------------------------
class PatientSelectorWidget(QtWidgets.QWidget):
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

# -------------------------------
# Main: Setup Napari Viewer and Patient Selector
# -------------------------------
dataset_path = r"C:\Users\niman\OneDrive\Desktop\TER_IPC\src\cached_lynos\Benchmark"
patient_folders = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.startswith("Pat")])
print("Patient folders:", patient_folders)

# Create a napari viewer in 3D mode.
viewer = napari.Viewer(ndisplay=3)

# Create empty layers for CT and labels, specifying 3D-friendly settings.
ct_layer = viewer.add_image(
    np.zeros((1,1,1), dtype=np.float32),
    name="CT Volume",
    blending="additive",
    rendering="mip",  # 3D rendering mode
)

labels_layer = viewer.add_labels(
    np.zeros((1,1,1), dtype=np.uint8),
    name="Lymph Nodes",
    opacity=0.6,
    rendering="translucent", 
)

labels_layer.color = {0: 'black', 1: 'green', 2: 'red'}

def load_patient(folder):
    print("Loading patient from folder:", folder)
    ct_data, ct_affine, new_seg, summary = process_patient(folder, cancer_thresh=15.0)

    ct_layer.data = ct_data

    ct_min, ct_max = float(np.min(ct_data)), float(np.max(ct_data))
    ct_layer.contrast_limits = (ct_min, ct_max)
    

    labels_layer.data = new_seg
    

    viewer.reset_view()
    
 
    print(f"Summary for {os.path.basename(folder)}:")
    for comp in summary:
        print(f"  Component {comp['component']}: {comp['short_axis_mm']:.1f} mm -> {comp['classification']}, center={comp['center_world']}")


selector = PatientSelectorWidget(patient_folders, load_patient)
viewer.window.add_dock_widget(selector, area='right')


napari.run()
