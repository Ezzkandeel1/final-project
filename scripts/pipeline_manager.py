# pipeline_manager.py
"""
Master Orchestrator Script for Multiclass Drone Detection.
Consolidates dataset consolidation (RGB/Thermal), modality tracking,
class remapping, and transfer learning with frozen backbone.
"""

import sys
import os
import argparse
import shutil
import random
import time
import yaml
import hashlib
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

# ------------------------------------------------------------------------------
#  DATASET CONFIGURATION
# ------------------------------------------------------------------------------
DATASETS_CONFIG = {
    "Anti-UAV.v1i.yolov11": {
        "modality": "thermal",
        "names": {0: 0} # drone -> Drone
    },
    "drone_night.v1i.yolov11": {
        "modality": "thermal",
        "names": {0: 0, 1: 0} # 0 -> Drone, drone -> Drone
    },
    "Thermal_drone_dataset.v1i.yolov11": {
        "modality": "thermal",
        "names": {0: 0, 1: 0} # DJI-Phantom -> Drone, Drone -> Drone
    },
    "Drone detection.v1i.yolov11": {
        "modality": "rgb",
        "names": {0: 1, 1: 0, 2: 2} # Bird -> Bird, Drone -> Drone, Plane -> Plane
    },
    "drone": {
        "modality": "rgb",
        "names": {0: 2, 1: 0} # AirPlane -> Plane, Drone -> Drone, Helicopter -> skip (2)
    },
    "Drone-detection-dataset-master-yolo/visible": {
        "modality": "rgb",
        "names": {0: 0, 1: 1, 2: 2} # Drone -> Drone, Bird -> Bird, Plane -> Plane
    },
    "Drone-detection-dataset-master-yolo/ir": {
        "modality": "thermal",
        "names": {0: 0, 1: 1, 2: 2} # Drone -> Drone, Bird -> Bird, Plane -> Plane
    },
    "Video_Project_2_override": {
        "modality": "rgb",
        "names": {0: 0} # Drone -> Drone
    },
    "hard_negative/Birds.v1i.yolov11": {
        "modality": "rgb",
        "names": {0: 1} # - bird -> Bird
    },
    "hard_negative/planes.v1i.yolov11": {
        "modality": "rgb",
        "names": {0: 2, 7: 1, 31: 3} # airplane -> Plane, bird -> Bird, kite -> Kite
    }
}

OUTPUT_DIR = "merged_multiclass_dataset"
VAL_RATIO = 0.15
TEST_RATIO = 0.05
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def get_file_signature(path: Path) -> str:
    return f"{path.name}_{path.stat().st_size}"

def find_images_and_labels(dataset_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs = []
    for split in ["train", "valid", "val", "test"]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if is_image(img):
                lbl = lbl_dir / (img.stem + ".txt")
                pairs.append((img, lbl if lbl.exists() else None))
    return pairs

def prep_dataset(mock_mode=False):
    random.seed(SEED)
    print("\n" + "=" * 60)
    print("  [PREP] CONSOLIDATING MULTICLASS DRONE DETECTION DATASET")
    print("=" * 60)

    out = Path(OUTPUT_DIR)
    if out.exists():
        print(f"\n[INFO] Removing previous output folder: {out}")
        shutil.rmtree(out)

    for split in ["train", "valid", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    all_samples = []
    seen_sigs = set()
    duplicates = 0

    print("\n[INFO] Scanning source datasets...")
    for ds_rel, cfg in DATASETS_CONFIG.items():
        ds_path = Path(ds_rel)
        if not ds_path.exists():
            print(f"   [SKIP] Dataset not found: {ds_rel}")
            continue

        pairs = find_images_and_labels(ds_path)
        
        # In mock mode, only take 6 images from each source dataset to make it super fast
        if mock_mode:
            pairs = pairs[:6]

        added_count = 0
        for img, lbl in pairs:
            sig = get_file_signature(img)
            if sig in seen_sigs:
                duplicates += 1
                continue
            seen_sigs.add(sig)
            all_samples.append({
                "img": img,
                "lbl": lbl,
                "modality": cfg["modality"],
                "mapping": cfg["names"],
                "source": ds_path.name
            })
            added_count += 1

        print(f"   [OK] {ds_path.name:36s} | Modality: {cfg['modality']:7s} | Added: {added_count}")

    print(f"\n[INFO] Total unique images collected: {len(all_samples)}")
    if duplicates > 0:
        print(f"[INFO] Duplicate files skipped: {duplicates}")

    # Shuffle and partition
    random.shuffle(all_samples)
    n = len(all_samples)
    
    if mock_mode:
        # For mock training, put 8 samples in train, 2 in val, 2 in test
        splits = {
            "train": all_samples[:8],
            "valid": all_samples[8:10],
            "test": all_samples[10:]
        }
    else:
        n_test = int(n * TEST_RATIO)
        n_val = int(n * VAL_RATIO)
        n_train = n - n_val - n_test
        splits = {
            "train": all_samples[:n_train],
            "valid": all_samples[n_train: n_train + n_val],
            "test": all_samples[n_train + n_val:]
        }

    global_class_counts = Counter()
    split_modality_counts = {"train": Counter(), "valid": Counter(), "test": Counter()}

    print("\n[INFO] Writing image files and labels...")
    for split_name, samples in splits.items():
        print(f"   Writing {split_name} split ({len(samples)} samples)...")
        for i, sample in enumerate(samples):
            img_path = sample["img"]
            lbl_path = sample["lbl"]
            modality = sample["modality"]
            mapping = sample["mapping"]
            source = sample["source"]

            clean_src = source.replace(".", "_").replace(" ", "_")
            unique_name = f"{modality}_{clean_src}_{img_path.stem}"
            suffix = img_path.suffix

            dst_img = out / split_name / "images" / f"{unique_name}{suffix}"
            dst_lbl = out / split_name / "labels" / f"{unique_name}.txt"

            # 15% probability of thermal simulation for RGB training images
            simulate_thermal = False
            if modality == "rgb" and split_name == "train" and random.random() < 0.15 and not mock_mode:
                simulate_thermal = True

            # Standardize / Convert / Save image
            # Grayscale images or simulated thermal require cv2 processing
            if modality == "thermal" or simulate_thermal:
                img = cv2.imread(str(img_path))
                if img is None:
                    shutil.copy2(img_path, dst_img)
                else:
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        # 1-channel to 3-channel replication
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    if simulate_thermal:
                        # Thermal signature simulation
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        min_val, max_val = random.randint(0, 40), random.randint(210, 255)
                        g_min, g_max = gray.min(), gray.max()
                        if g_max > g_min:
                            gray = np.clip((gray - g_min) / (g_max - g_min) * (max_val - min_val) + min_val, 0, 255).astype(np.uint8)
                        if random.random() < 0.3:
                            gray = 255 - gray
                        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    
                    cv2.imwrite(str(dst_img), img)
            else:
                shutil.copy2(img_path, dst_img)

            split_modality_counts[split_name][modality] += 1

            # Build label lines
            new_lines = []
            if lbl_path and lbl_path.exists():
                content = lbl_path.read_text().strip()
                for line in content.splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        orig_cls = int(parts[0])
                        if orig_cls in mapping:
                            target_cls = mapping[orig_cls]
                            parts[0] = str(target_cls)
                            new_lines.append(" ".join(parts))
                            global_class_counts[target_cls] += 1
                    except ValueError:
                        continue

            # Handles unannotated background images safely by creating empty label files
            dst_lbl.write_text("\n".join(new_lines) + "\n" if new_lines else "")

    # Write data_multiclass.yaml
    data_cfg = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
        "names": {
            0: "Drone",
            1: "Bird",
            2: "Plane",
            3: "Kite"
        }
    }
    yaml_path = out / "data_multiclass.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_cfg, f, default_flow_style=False, sort_keys=False)

    print("\n[OK] Dataset consolidation complete!")
    print(f"     Config file: {yaml_path.resolve()}")
    for split_name, counts in split_modality_counts.items():
        rgb_c = counts.get("rgb", 0)
        therm_c = counts.get("thermal", 0)
        tot = rgb_c + therm_c
        pct_rgb = (rgb_c / tot * 100) if tot > 0 else 0
        pct_therm = (therm_c / tot * 100) if tot > 0 else 0
        print(f"     {split_name.capitalize():6s} : {tot:>5} total | RGB: {rgb_c:>5} ({pct_rgb:.1f}%) | Thermal: {therm_c:>5} ({pct_therm:.1f}%)")
    
    print("     Class labels bounding boxes count:")
    class_names = ["Drone", "Bird", "Plane", "Kite"]
    for i, name in enumerate(class_names):
        print(f"      - {name:6s}: {global_class_counts[i]} boxes")

def train_model(mock_mode=False, base_model="yolo11x.pt", epochs=80, batch_size=-1, imgsz=640, freeze=10):
    print("\n" + "=" * 60)
    print("  [TRAIN] INITIALIZING YOLOv11 TRANSFER LEARNING STAGE")
    print("=" * 60)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Training Device : {device}")

    here = Path(__file__).resolve().parent
    yaml_path = here / OUTPUT_DIR / "data_multiclass.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] Dataset configuration YAML not found at: {yaml_path}")
        print("        Please run the preparation step first.")
        sys.exit(1)

    # Mock parameters override
    if mock_mode:
        print("[INFO] Mock mode enabled: running 1 epoch on mock subset.")
        epochs = 1
        batch_size = 2
        imgsz = 320
    else:
        if batch_size == -1:
            # Auto-configure batch size based on VRAM
            if device == 0:
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram >= 16: batch_size = 32
                elif vram >= 8: batch_size = 16   # RTX 3060 Ti: only used 3.9GB at batch 8
                elif vram >= 6: batch_size = 8
                else: batch_size = 4
            else:
                batch_size = 2

    print(f"[INFO] Settings: Epochs={epochs}, BatchSize={batch_size}, ImageSize={imgsz}")

    # Load weights
    print(f"\n[INFO] Loading base model: {base_model} ...")
    model = YOLO(base_model)

    # Lock backbone parameters programmatically to verify
    frozen_p, trainable_p = 0, 0
    if freeze and freeze > 0:
        print(f"[INFO] Freezing first {freeze} backbone layers...")
        freeze_prefixes = [f"model.{i}." for i in range(freeze)]
        for name, param in model.model.named_parameters():
            is_frozen = any(p in name for p in freeze_prefixes)
            if is_frozen:
                param.requires_grad = False
                frozen_p += param.numel()
            else:
                param.requires_grad = True
                trainable_p += param.numel()
    else:
        print("[INFO] Training all layers (no freezing)...")
        for name, param in model.model.named_parameters():
            param.requires_grad = True
            trainable_p += param.numel()
    
    print(f"       Frozen params   : {frozen_p:,}")
    print(f"       Trainable params: {trainable_p:,}")

    run_project = str(here / "runs" / "detect")
    run_name = "consolidated_transfer_yolo11"
    if mock_mode:
        run_name += "_mock"

    print(f"\n[INFO] Starting training. Output folder: {run_project}/{run_name}")
    try:
        model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=run_project,
            name=run_name,
            exist_ok=True,
            patience=15 if not mock_mode else 1,
            device=device,
            freeze=freeze,  # Native parameter locks the first N modules (backbone)
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.001,
            augment=True,
            mosaic=1.0,
            fliplr=0.5,
            flipud=0.2,
            workers=0,  # Safe for Windows multiprocess lockups
            val=True
        )
        print("\n[OK] Training completed successfully!")
        best_pt = Path(run_project) / run_name / "weights" / "best.pt"
        print(f"     Trained model weights: {best_pt.resolve()}")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrator for drone detection pipeline.")
    parser.add_argument("--mock", action="store_true", help="Run a quick 1-epoch mock pipeline check.")
    parser.add_argument("--skip-prep", action="store_true", help="Skip dataset preparation stage.")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Pretrained weights to use.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100).")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640).")
    parser.add_argument("--freeze", type=int, default=10, help="Number of backbone layers to freeze (default: 10, use 0 to unfreeze all).")
    args = parser.parse_args()

    # Step 1: Prep Dataset
    if not args.skip_prep:
        prep_dataset(mock_mode=args.mock)
    else:
        print("\n[INFO] Skipping dataset preparation stage.")

    # Step 2: Train Model
    # Note: If we run mock, we use a smaller local model like yolo11m.pt or yolo26n.pt to avoid 
    # downloading 100+ MB file over the network if it's not present.
    # If the user specifies yolo11x.pt, we will use it, but for mock, let's fall back to yolo11m.pt 
    # if yolo11x.pt is not downloaded.
    model_to_use = args.model
    if args.mock and model_to_use == "yolo11x.pt":
        # Check if local yolo11m.pt exists
        if Path("yolo11m.pt").exists():
            model_to_use = "yolo11m.pt"
        elif Path("yolo26n.pt").exists():
            model_to_use = "yolo26n.pt"

    train_model(
        mock_mode=args.mock,
        base_model=model_to_use,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        freeze=args.freeze
    )
