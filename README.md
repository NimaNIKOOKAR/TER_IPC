# Automatic Detection, 3D Segmentation, and AI-Assisted Characterization of Pathological Lymph Node Formations in Thoraco-Abdomino-Pelvic CT Scans for Onco-Hematology Patients


## Project Description

### Context and Problem Statement
Artificial intelligence (AI) has demonstrated significant potential in the automatic detection of tumor lesions in CT scans. Several commercial AI-driven solutions for detecting and segmenting pulmonary and hepatic lesions are already used in clinical practice.

The ultimate goal of AI-assisted imaging interpretation is to detect, segment, and characterize all normal and pathological structures, thereby saving time, optimizing workflow for radiologists, and enhancing efficiency.

A substantial portion of initial extension assessments and follow-up examinations in onco-hematology relies on cervico-thoraco-abdomino-pelvic CT scans. The presence or absence of pathological lymph nodes in initial and follow-up imaging helps stage diseases and adapt therapeutic strategies throughout the patient’s journey.

Currently, no AI-based solution is available that provides simultaneous detection, segmentation, and characterization of lymph node formations.

### Research Context
Recent research on lymph nodes using machine learning primarily focuses on:
- Lymph node localization.
- Dichotomous classification (normal vs. pathological).  
**Reference:**
- Bedrikovetski S et al., "Artificial intelligence for pre-operative lymph node staging in colorectal cancer: A systematic review and meta-analysis." BMC Cancer (2021). DOI: [10.1186/s12885-021-08773-w](https://doi.org/10.1186/s12885-021-08773-w)

## Objectives
The project aims to develop and validate an AI algorithm capable of:
1. **Automatic detection**, **segmentation**, and **measurement** of lymph nodes larger than **15mm**.
2. **Characterization of lymph nodes** (especially challenging for those between **10mm and 15mm**).
3. **Industrial partnership opportunities** if results prove positive.

## Data and Constraints
### Data Sources
- **Training Data:**
  - **Anonymized imaging data from IPC** (Thoraco-Abdomino-Pelvic CT scans of 50 onco-hematology patients, ~450 images per exam in DICOM or TIFF format).
  - Includes **confirmed normal and pathological lymph nodes**, verified by **PET-CT** or **histopathological analysis**, and segmented by radiologists.
- **External Validation Data:**
  - **[CT Lymph Nodes dataset](https://www.cancerimagingarchive.net/collection/ct-lymph-nodes/)** (publicly available)
  - **[LyNoS dataset](https://github.com/raidionics/LyNoS)** (publicly available)


### Compliance Requirements
- **Approval from GSPC**
- **MR004 regulatory compliance**
- **Registration on MesDonnées.com (DRCI of IPC)**

## Technical Approach
The project will involve:
1. **Data Preprocessing**:
   - Conversion of DICOM/TIFF images into a suitable format for deep learning models.
   - Data augmentation and normalization.
2. **Model Development**:
   - Implementation of **3D convolutional neural networks (CNNs)** for segmentation.
   - Incorporation of **anatomical priors** to enhance detection accuracy.
3. **Model Training & Validation**:
   - Use of **50-patient IPC dataset** for initial training.
   - External validation on **publicly available datasets**.
4. **Evaluation Metrics**:
   - **Segmentation performance** (Dice similarity coefficient, Intersection-over-Union (IoU)).
   - **Detection accuracy** (Sensitivity, Specificity, AUC-ROC score).

## References
- **Bouget D et al. (2022).** "Mediastinal lymph nodes segmentation using 3D convolutional neural network ensembles and anatomical priors guiding." *Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization.* DOI: [10.1080/21681163.2022.2043778](https://doi.org/10.1080/21681163.2022.2043778)


## How to Run the Project
1. **Clone Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Prepare Data:**
   - Ensure **DICOM/TIFF images** are placed in the correct dataset folder.
   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```
3. **Train the Model:**
   ```bash
   python train.py
   ```


## Contact
For further inquiries, please reach out to the project team.
nimanikoukar.apple@gmail.com
