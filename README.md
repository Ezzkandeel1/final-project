# Drone Detection — Multiclass Transfer Learning & Active Learning Pipeline

A robust, production-ready **YOLOv11-based multiclass object detection pipeline** designed to detect and classify **Drones**, **Birds**, **Planes**, and **Kites** across both **RGB** and **Thermal (Infrared)** modalities. 

This repository includes a complete active-learning and error-correction loop designed to eliminate classification bias (e.g., misclassifying close-up quadcopters as airplanes) using targeted pseudo-labeling.

---

## 📁 Repository Structure

```
final project/
│
├── scripts/                         # Core execution scripts
│   ├── pipeline_manager.py          # ⭐ Master orchestrator (dataset prep + training)
│   ├── prepare_multiclass_dataset.py # Consolidates, remaps, and splits all source datasets
│   ├── extract_matlab_dataset.py    # Decodes MATLAB groundTruth MCOS objects into YOLO annotations
│   ├── generate_pseudo_labels.py    # Active learning tool to extract and correct misclassified frames
│   ├── run_inference.py             # Real-time webcam and video file inference visualizer
│   ├── val_transfer.py              # Modality-split (RGB / Thermal) validation script
│   └── train_transfer.py            # Custom training loop helper
│
├── configs/
│   └── data_multiclass.yaml         # YOLOv11 dataset config (classes, paths)
│
├── models/
│   └── best.pt                      # 🏆 Pre-trained / Fine-tuned multiclass weights (YOLOv11s)
│
├── results/
│   ├── confusion_matrices/          # Evaluation results (Combined, RGB, Thermal matrices)
│   └── training_plots/              # YOLO loss curves, PR curves, and prediction previews
│
├── docs/
│   └── ARCHITECTURE.md              # Deep-dive into design decisions & normalization rules
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Ensures large datasets, caches, and logs are not committed
└── README.md                        # This file
```

---

## 🎯 Target Classes & Taxonomy

Interactive visualizations and bounding boxes follow a consistent class mapping and color palette:

| ID | Class | Colour in Inference | Description |
|:--:|:-----:|:--------------------|:------------|
| 0  | Drone  | 🔴 Red (BGR: `[0, 0, 255]`)  | Quadcopters, hexacopters, fixed-wing UAVs |
| 1  | Bird   | 🟢 Green (BGR: `[0, 255, 0]`) | Flying birds, flocks |
| 2  | Plane  | 🔵 Blue (BGR: `[255, 0, 0]`)  | Commercial, military, and general aviation aircraft |
| 3  | Kite   | 🟠 Orange (BGR: `[0, 165, 255]`)| Recreatonal kites, wind-sails |

---

## 🗂️ Dataset Sources & Consolidation

The pipeline integrates and standardizes **8 heterogeneous datasets** into a single unified schema:

| Dataset | Modality | Original Classes | Target Classes |
|:--------|:--------:|:-----------------|:---------------|
| **Matlab Visible Dataset** | RGB | DRONE, BIRD, AIRPLANE, HELICOPTER | Drone, Bird, Plane |
| **Matlab Thermal Dataset** | Thermal | DRONE, BIRD, AIRPLANE, HELICOPTER | Drone, Bird, Plane |
| **Anti-UAV.v1i.yolov11** | Thermal | drone | Drone |
| **drone_night.v1i.yolov11** | Thermal | drone | Drone |
| **Thermal_drone_dataset.v1i.yolov11**| Thermal | DJI-Phantom, Drone | Drone |
| **Drone detection.v1i.yolov11** | RGB | Bird, Drone, Plane | Bird, Drone, Plane |
| **drone** (Roboflow) | RGB | AirPlane, Drone | Plane, Drone |
| **hard_negative/Birds.v1i.yolov11** | RGB | bird | Bird |
| **hard_negative/planes.v1i.yolov11** | RGB | airplane, bird, kite | Plane, Bird, Kite |
| **Video_Project_2_override** | RGB | Drone (user-specific domain) | Drone (high scale) |

---

## ⚙️ Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Place Source Datasets
Place the raw dataset folders matching the paths configured in `prepare_multiclass_dataset.py` (next to the `scripts/` directory).

---

## 🚀 How to Run the Pipeline

### 1. Rebuild and Consolidate Datasets
Generates a clean train/val/test split, performs offline thermal simulations for domain bridging, and creates `data_multiclass.yaml`:
```bash
python scripts/prepare_multiclass_dataset.py
```

### 2. Run Transfer Learning (Fine-Tuning)
Train the model with a fully **unfrozen backbone** (`--freeze 0`) to allow YOLOv11s to update its feature extractor and overcome pre-trained COCO classification biases:
```bash
python scripts/pipeline_manager.py --skip-prep --epochs 80 --model yolo11s.pt --imgsz 416 --batch 32 --freeze 0
```

### 3. Modality-Specific Evaluation
Evaluate performance separately on the **RGB**, **Thermal**, and **Combined** test splits to generate precision/recall statistics and custom confusion matrices:
```bash
python scripts/val_transfer.py
```

---

## 🔄 Active Learning & Bias Correction Loop

When a model misclassifies a close-up or profile drone as a plane (due to COCO wing-feature activation), use the pseudo-labeling tool:

1. **Extract and Pseudo-label**: Run the script to extract frames from the test video where objects are localized, but override their labels to **Drone (class 0)**:
   ```bash
   python scripts/generate_pseudo_labels.py
   ```
2. **Re-consolidate**: Run `prepare_multiclass_dataset.py` to merge the generated `Video_Project_2_override` dataset.
3. **Fine-tune**: Run a quick 10-epoch training run starting from your best weights:
   ```bash
   python scripts/pipeline_manager.py --skip-prep --epochs 10 --model models/best.pt --imgsz 416 --batch 32 --freeze 0
   ```

---

## 🔍 Running Inference

Run inference on any image, video file, or live webcam. The script automatically highlights objects using class-specific colors:

### Live Webcam Stream
```bash
python scripts/run_inference.py --source 0
```

### Video File (Saves Output)
```bash
python scripts/run_inference.py --source path/to/video.mp4 --save
```

### Headless / CLI-only Mode (No GUI window)
```bash
python scripts/run_inference.py --source path/to/video.mp4 --save --headless
```
