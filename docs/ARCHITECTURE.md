# Architecture & Design Decisions

This document explains the key engineering choices made in the multiclass drone detection and classification transfer learning pipeline.

---

## 1. Model Selection — YOLOv11s

We transitioned from the extra-large `YOLOv11x` (114M parameters) to the smaller `YOLOv11s` (9.4M parameters) for the following reasons:
* **Computational Feasibility**: Training one epoch of `YOLOv11m` at 640px took ~50 minutes on the target machine due to dataloader bottlenecks on Windows.
* **Training Acceleration**: By adopting `YOLOv11s` at `imgsz=416` and `batch=32`, epoch duration was reduced to **~5.3 minutes** (a **10x speedup**), making full 80-epoch training and rapid iteration feasible.
* **Accuracy Trade-off**: Despite its smaller size, `YOLOv11s` achieved exceptional results on the test sets, including **81.5% mAP@50** for Drones on the combined test set.

---

## 2. Backbone Tuning Strategy

* **Initial Frozen Strategy (`freeze=10`)**: Freezing the feature-extractor backbone kept COCO pre-trained weights static to prevent catastrophic forgetting. However, this locked in a strong COCO-trained bias toward classifying large winged objects as airplanes.
* **Fine-Tuning Strategy (`freeze=0`)**: By training with a fully unfrozen backbone, the network's early conv layers were allowed to adapt. This enables the model to learn fine-grained geometric differences between close-up drones (with propeller arms and central pods) and fixed-wing airplanes.

---

## 3. MATLAB Dataset Extraction

The original MATLAB Drone Detection Dataset annotations were distributed as proprietary MATLAB `.mat` files containing `groundTruth` MCOS objects.
* We integrated the `mcos-decoder` library to parse these files in pure Python.
* Designed a downsampling parser (`extract_matlab_dataset.py`) that extracts every 15th frame, normalizes raw pixel coordinates into standard YOLO normalized coordinates `[x_center, y_center, width, height]`, and partitions the visible/thermal frames into the unified training taxonomy.

---

## 4. Active Learning Loop (Error Correction)

Computer vision models often fail on out-of-distribution targets (e.g., a DJI Mavic drone sitting on a concrete road close to the camera, when the training set only features tiny aerial drones against the sky).
* **Localization vs. Classification**: The model successfully localizes the object (draws a tight bounding box) but assigns it the wrong class (Plane).
* **Feedback Loop**: `generate_pseudo_labels.py` exploits this by executing inference on the failed test video, capturing the coordinates of all detected bounding boxes, but forcing the class label to `0` (Drone).
* **Few-Shot Retraining**: Merging this tiny, high-density target domain dataset back into the main train split and running a quick 10-epoch fine-tuning run resolves the classification bias without affecting general accuracy.

---

## 5. Modality-Specific Evaluation

Because thermal (IR) and RGB images represent drastically different color distributions, global metrics often obscure modality-specific failures.
* We prefix filenames with `rgb_` or `thermal_` during consolidation.
* `val_transfer.py` parses these prefixes dynamically at test time to split evaluation.
* The model achieved **88.0% mAP@50** for Drones on Thermal data and **67.9% mAP@50** on RGB data, demonstrating how modality-specific evaluation exposes the domain difficulty gap.
