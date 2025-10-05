# Swin UNETR — BTCV Multi-Organ 3D Segmentation (Pretrained + Custom Upgrades)

This repository/notebook provides an end-to-end **3D abdominal CT multi-organ segmentation** pipeline based on the pretrained **Swin UNETR** MONAI bundle.  
It includes **custom upgrades** for inference, volumetry (CSV export), and morphometrics.

- **Base model**: `monai_swin_unetr_btcv_segmentation` (pretrained on BTCV)
- **Provider / Training**: **MONAI Consortium** (bundle distributed via NVIDIA NGC)
- **Upgrades & integration**: **Abdulrahman HAMDI**

---

## ✨ What’s New (Upgrades)

1. **Custom inference config** (`configs/inference_custom.json`) to simplify running on Kaggle/Colab and custom paths.
2. **Volumetry pipeline** (`btcv-3d-volume-pipeline.py`)
   - Batch scans your predicted masks, reads spacing from NIfTI headers, and exports per-organ **volumes (mm³/mL)** to a CSV.
3. **Morphometrics** (`btcv_morphometrics.py`)
   - Advanced shape metrics (e.g., **surface area**, **Hausdorff distance**, **surface Dice**, **sphericity**), and utilities for surface voxels.
4. **Visualization utilities**
   - Quick 2D/3D previews using Matplotlib/Plotly + skimage for overlays and meshes.
5. **Isotropic resampling & surface tools**
   - Helper functions such as `resample_isotropic_labels()` and `surface_area_mm2()` for fair, resolution-agnostic measurements.

---

## 📦 Requirements

- Python 3.9–3.11 (Colab/Kaggle OK)
- GPU recommended
- Core packages:
  - `monai[nibabel,tqdm]==1.3.0`, `torch`/`torchvision` (per your runtime), `nibabel`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `plotly`, `fire`  
- Optional acceleration:
  - TensorRT + torch-tensorrt (A100/L4/T4 depending on environment)

> **Important**: You must include the two helper scripts in your environment:
> - `btcv-3d-volume-pipeline.py` — **volumetry & CSV export** (spacing-aware)
> - `btcv_morphometrics.py` — **advanced metrics/morphometrics** (surface distance, area, etc.)

Place them in a folder you can `sys.path.append(...)`, or install as a package.

---

## 📁 Data (BTCV)

- Dataset: **BTCV** (Beyond the Cranial Vault)
- Expected folder structure (after unzip/reorg):
  ```
  RawData/
    imagesTr/
    labelsTr/
    imagesTs/
  ```
- Model inputs: **single-channel CT**; patch size typically `96×96×96`
- Output labels (14):
  ```
  0 Background
  1 Spleen
  2 Right Kidney
  3 Left Kidney
  4 Gallbladder
  5 Esophagus
  6 Liver
  7 Stomach
  8 Aorta
  9 IVC
  10 Portal & Splenic Veins
  11 Pancreas
  12 Right Adrenal Gland
  13 Left Adrenal Gland
  ```

---

## 🚀 Quickstart (Kaggle/Colab)

### 1) Install dependencies
```bash
!pip install "monai[nibabel,tqdm]==1.3.0" fire
!pip install -q scipy scikit-image matplotlib plotly nibabel
```

### 2) Make helper modules available
Place the files and append their directory to `sys.path`:
```python
import sys
sys.path.append("/kaggle/input/btcv-3d-volume-pipeline")   # adjust path
sys.path.append("/kaggle/input/btcv-morphometrics")        # adjust path
from btcv_3d_volume_pipeline import process_folder_to_csv
import btcv_morphometrics
```

### 3) Run inference (MONAI bundle CLI)
Prepare `configs/inference_custom.json` with your **weights path**, **input**, and **output** directories, then:
```bash
!python -m monai.bundle run --config_file configs/inference_custom.json
```
Outputs typically go under:
```
.../swin_unetr_btcv_segmentation/eval/<run_id>/
```

### 4) Check outputs
```bash
!ls /kaggle/working/swin_unetr_btcv_segmentation/eval/
```

### 5) Export organ volumes to CSV
```python
from btcv_3d_volume_pipeline import process_folder_to_csv
csv_path = process_folder_to_csv(
    "/kaggle/working/swin_unetr_btcv_segmentation/eval/*/*_trans.nii.gz",
    out_csv="organ_volumes.csv"
)
print("Saved:", csv_path)
```

### 6) (Optional) Morphometrics / validation metrics
```python
import btcv_morphometrics as mm

# Example stubs (actual APIs depend on your module):
# hd95 = mm.hausdorff95(pred_mask, gt_mask, spacing)
# sds  = mm.surface_dice(pred_mask, gt_mask, spacing, tol_mm=2.0)
# sa   = mm.surface_area_mm2(binary_mask, spacing)
```

### 7) Visualize predictions
```python
import nibabel as nib, matplotlib.pyplot as plt
pred_path = "/kaggle/working/swin_unetr_btcv_segmentation/eval/<run_id>/pred_001.nii.gz"
pred = nib.load(pred_path).get_fdata()
slice_idx = pred.shape[2] // 2
plt.imshow(pred[:, :, slice_idx], cmap="nipy_spectral"); plt.title("Pred mask - mid slice"); plt.axis("off")
```

---

## ⚙️ TensorRT (optional)
If you’ve exported a TensorRT engine and prepared `configs/inference_trt.json`, you can combine configs:
```bash
!python -m monai.bundle run --config_file "['configs/inference_custom.json', 'configs/inference_trt.json']"
```

---

## 🔧 Helper Scripts (What they do)

### `btcv-3d-volume-pipeline.py`
- **Purpose**: Batch-process predicted NIfTI masks and compute **per-organ volumes** using voxel spacing.
- **Key function**: `process_folder_to_csv(glob_pattern, out_csv="organ_volumes.csv")`
- **Output**: A tidy CSV with rows = cases, columns = organ volumes (mm³ / mL).  
- **Extras**: Utilities for isotropic resampling and label filtering (if included in your version).

### `btcv_morphometrics.py`
- **Purpose**: Compute **morphometric** and **surface-based** validation metrics:
  - Surface area (mm²), Hausdorff (e.g., HD95), surface Dice, sphericity, etc.
- **Helpers**: Surface voxel extraction, distance transforms, mesh-based area via skimage.

> Ensure both files are accessible in your runtime and correctly imported before running volumetry/metrics.

---

## 📝 Tips & Conventions

- **Paths**: Kaggle uses `/kaggle/input/` and `/kaggle/working/`; Colab typically uses `/content/`.
- **OOM issues**: Reduce `cache_rate`, batch size, or patch size; use AMP.
- **File types**: Your pipeline supports both `.nii` and `.nii.gz`. Keep it consistent across steps.
- **Isotropic resampling**: Use nearest-neighbor for **labels** to preserve integer classes.
- **Reproducibility**: Fix seeds where possible; log versions of `torch`, `monai`, `nibabel`, etc.

---

## 🙌 Credits

- **Pretrained model & bundle**: **MONAI Consortium** (distributed on **NVIDIA NGC**).  
- **Upgrades, integration & documentation**: **Abdulrahman HAMDI**.
