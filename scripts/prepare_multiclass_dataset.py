# prepare_multiclass_dataset.py
"""
Merge and preprocess multiple datasets for Drone, Bird, Plane, and Kite classification.
Standardizes labels and splits the data into train/val/test splits.
Tracks image modality (RGB vs Thermal) using filename prefixes.
"""

import shutil
import random
import yaml
import hashlib
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------

# Input datasets with their expected class names and source modalities
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
        # 60 classes total. We map:
        # airplane (0) -> Plane (2)
        # bird (7) -> Bird (1)
        # kite (31) -> Kite (3)
        # others are ignored (not copied to target label)
        "names": {0: 2, 7: 1, 31: 3}
    }
}

# Target output directory
OUTPUT_DIR = "merged_multiclass_dataset"

# Splitting ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.05
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

def file_hash(path: Path) -> str:
    # Fast size-and-name signature instead of md5 to avoid reading 15k+ files
    return f"{path.name}_{path.stat().st_size}"

def find_images_and_labels(dataset_dir: Path) -> list[tuple[Path, Path | None]]:
    """Finds all (image, label) pairs in a YOLO export structure."""
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

def main():
    random.seed(SEED)
    print("\n" + "=" * 60)
    print("  [MERGE] MULTICLASS DRONE/BIRD/PLANE/KITE DATASET PREPARATION")
    print("=" * 60)

    out = Path(OUTPUT_DIR)
    if out.exists():
        print(f"\n[WARN] Removing previous output dataset: {out}")
        shutil.rmtree(out)

    for split in ["train", "valid", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all samples: list of dicts containing image_path, label_path, modality, mappings
    all_collected_samples = []
    seen_hashes = set()
    duplicates_removed = 0
    total_found = 0

    print("\n[INFO] Scanning datasets...")
    for ds_rel_path, cfg in DATASETS_CONFIG.items():
        ds_path = Path(ds_rel_path)
        if not ds_path.exists():
            print(f"   [SKIP] Not found: {ds_rel_path}")
            continue

        modality = cfg["modality"]
        class_mapping = cfg["names"]
        pairs = find_images_and_labels(ds_path)
        total_found += len(pairs)

        valid_pairs_count = 0
        for img, lbl in pairs:
            try:
                h = file_hash(img)
                if h in seen_hashes:
                    duplicates_removed += 1
                    continue
                seen_hashes.add(h)
                all_collected_samples.append({
                    "img": img,
                    "lbl": lbl,
                    "modality": modality,
                    "mapping": class_mapping,
                    "source": ds_path.name
                })
                valid_pairs_count += 1
            except Exception as e:
                print(f"Error hashing {img}: {e}")

        print(f"   [OK] {ds_path.name:40s} | Modality: {modality:7s} | Found: {len(pairs):>5} | Added: {valid_pairs_count:>5}")

    print(f"\n   Total Raw Images: {total_found}")
    print(f"   Duplicate Images Removed: {duplicates_removed}")
    print(f"   Unique Images Kept: {len(all_collected_samples)}")

    # Shuffle and split globally to ensure RGB/Thermal and classes are evenly distributed
    random.shuffle(all_collected_samples)

    n = len(all_collected_samples)
    n_test = int(n * TEST_RATIO)
    n_val = int(n * VAL_RATIO)
    n_train = n - n_val - n_test

    splits = {
        "train": all_collected_samples[:n_train],
        "valid": all_collected_samples[n_train: n_train + n_val],
        "test": all_collected_samples[n_train + n_val:]
    }

    # Track metrics
    global_class_counts = Counter()
    modality_counts = {"train": Counter(), "valid": Counter(), "test": Counter()}

    print("\n[INFO] Writing output dataset splits...")
    for split_name, samples in splits.items():
        print(f"\n   Writing {split_name}/ ({len(samples)} images) ...")
        for i, sample in enumerate(samples):
            img_path = sample["img"]
            lbl_path = sample["lbl"]
            modality = sample["modality"]
            mapping = sample["mapping"]
            source = sample["source"]

            # Standardize filename to include modality and source name to prevent collisions
            clean_source_name = source.replace(".", "_").replace(" ", "_")
            unique_name = f"{modality}_{clean_source_name}_{img_path.stem}"
            suffix = img_path.suffix

            dst_img = out / split_name / "images" / f"{unique_name}{suffix}"
            dst_lbl = out / split_name / "labels" / f"{unique_name}.txt"

            # Determine if we should simulate a thermal image from RGB (15% chance in train split)
            simulate_thermal = False
            if modality == "rgb" and split_name == "train" and random.random() < 0.15:
                simulate_thermal = True

            # If it's a thermal dataset or we want to simulate thermal, we process it
            if modality == "thermal" or simulate_thermal:
                img = cv2.imread(str(img_path))
                if img is None:
                    # Fallback to copy if read fails
                    shutil.copy2(img_path, dst_img)
                else:
                    # Check channels
                    if len(img.shape) == 2 or img.shape[2] == 1:
                        # Grayscale: repeat to 3-channel
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
                    if simulate_thermal:
                        # Convert RGB to Grayscale
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        # Contrast stretching (min/max stretching)
                        min_val = random.randint(0, 40)
                        max_val = random.randint(210, 255)
                        g_min, g_max = gray.min(), gray.max()
                        if g_max > g_min:
                            gray = np.clip((gray - g_min) / (g_max - g_min) * (max_val - min_val) + min_val, 0, 255).astype(np.uint8)
                        # 30% chance of inverting hot/cold
                        if random.random() < 0.3:
                            gray = 255 - gray
                        # Convert back to 3-channel
                        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    
                    cv2.imwrite(str(dst_img), img)
            else:
                # Standard RGB: fast copy
                shutil.copy2(img_path, dst_img)

            modality_counts[split_name][modality] += 1

            # Process and write label file
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

            # Write the label file (create empty file if no valid boxes)
            dst_lbl.write_text("\n".join(new_lines) + "\n" if new_lines else "")

            if (i + 1) % 1000 == 0:
                print(f"      Processed {i+1}/{len(samples)} files...")

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

    print("\n" + "=" * 60)
    print("  [OK] DATASET MERGE COMPLETE!")
    print("=" * 60)
    print(f"   Output Folder: {out.resolve()}")
    print(f"   Config File  : {yaml_path.resolve()}")
    print("\n   [INFO] Split Breakdown:")
    for split_name, counts in modality_counts.items():
        rgb_c = counts.get("rgb", 0)
        therm_c = counts.get("thermal", 0)
        tot = rgb_c + therm_c
        print(f"      {split_name.capitalize():6s} : {tot:>5} total | RGB: {rgb_c:>5} ({rgb_c/tot*100:.1f}%) | Thermal: {therm_c:>5} ({therm_c/tot*100:.1f}%)")

    print("\n   [INFO] Class Distribution (Bounding Boxes):")
    class_names = ["Drone", "Bird", "Plane", "Kite"]
    for cls_idx, name in enumerate(class_names):
        print(f"      Class {cls_idx} ({name}): {global_class_counts[cls_idx]:>5} boxes")
    print()

if __name__ == "__main__":
    main()
