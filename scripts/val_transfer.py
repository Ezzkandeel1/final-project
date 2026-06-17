# val_transfer.py
"""
Evaluates the fine-tuned multiclass drone detection model.
Separates the test dataset into RGB and Thermal subsets based on filename prefixes,
runs validation on each modality, and outputs performance metrics and confusion matrices.
"""

import sys
import os
import argparse
import shutil
import time
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def print_banner():
    print("\n" + "=" * 60)
    print("  [EVAL] YOLOv11 MULTICLASS MODALITY-SPECIFIC EVALUATION")
    print("=" * 60)

def find_latest_model(here: Path) -> Path:
    """Finds the best.pt from the transfer learning runs."""
    search_root = here / "runs" / "detect"
    candidates = list(search_root.rglob("best.pt"))
    if not candidates:
        # Fallback to look in the workspace root for any runs folder
        search_root = here.parent / "runs" / "detect"
        candidates = list(search_root.rglob("best.pt"))
        
    if not candidates:
        print("[ERROR] No trained best.pt model found in runs/detect/.")
        print("        Please run train_transfer.py first.")
        sys.exit(1)
        
    # Sort by modification time, get latest
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"[INFO] Auto-detected latest model: {candidates[0].resolve()}")
    return candidates[0]

def create_split_yaml(base_dir: Path, image_paths: list, name: str) -> Path:
    """Creates a text file list of images and a corresponding data.yaml for YOLO val."""
    # Write image list file
    txt_path = base_dir / f"test_{name}.txt"
    with open(txt_path, "w") as f:
        for p in image_paths:
            f.write(str(p.resolve()) + "\n")
            
    # Write YAML config
    yaml_data = {
        "path": str(base_dir.resolve()),
        "train": "train/images",  # Placeholders, only val/test split is used
        "val": f"test_{name}.txt",
        "nc": 4,
        "names": {
            0: "Drone",
            1: "Bird",
            2: "Plane",
            3: "Kite"
        }
    }
    
    yaml_path = base_dir / f"data_val_{name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    return yaml_path

def print_performance_table(modality: str, metrics):
    """Parses and prints a beautiful performance metrics table."""
    print(f"\n   Metrics Summary for [{modality.upper()}]:")
    print(f"   {'-'*65}")
    print(f"   {'Class':12s} | {'Precision':10s} | {'Recall':10s} | {'mAP@50':10s} | {'mAP@50-95':10s}")
    print(f"   {'-'*65}")
    
    # Global metrics
    mp = metrics.box.mp
    mr = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    print(f"   {'ALL (mean)':12s} | {mp:10.4f} | {mr:10.4f} | {map50:10.4f} | {map5095:10.4f}")
    print(f"   {'-'*65}")
    
    # Class-specific metrics
    class_names = ["Drone", "Bird", "Plane", "Kite"]
    for i, name in enumerate(class_names):
        try:
            # box.class_result(i) returns [p, r, ap50, ap]
            res = metrics.box.class_result(i)
            p, r, ap50, ap = res
            print(f"   {name:12s} | {p:10.4f} | {r:10.4f} | {ap50:10.4f} | {ap:10.4f}")
        except Exception:
            print(f"   {name:12s} | {'N/A':10s} | {'N/A':10s} | {'N/A':10s} | {'N/A':10s}")
    print(f"   {'-'*65}")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned multiclass drone model.")
    parser.add_argument("--model", type=str, default=None, help="Path to best.pt weights (auto-detected if omitted).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS (default: 0.6).")
    parser.add_argument("--mock", action="store_true", help="Run a fast evaluation on 10 images.")
    parsed_args = parser.parse_args()

    here = Path(__file__).resolve().parent
    dataset_dir = here / "merged_multiclass_dataset"
    
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found at: {dataset_dir}")
        sys.exit(1)
        
    test_img_dir = dataset_dir / "test" / "images"
    if not test_img_dir.exists():
        print(f"[ERROR] Test split images folder not found at: {test_img_dir}")
        sys.exit(1)

    # Find model weights
    model_path = Path(parsed_args.model) if parsed_args.model else find_latest_model(here)
    if not model_path.exists():
        print(f"[ERROR] Model weights file not found at: {model_path}")
        sys.exit(1)

    # 1. Filter test set into RGB and Thermal categories
    all_test_imgs = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.jpeg")) + list(test_img_dir.glob("*.png"))
    
    if parsed_args.mock:
        all_test_imgs = all_test_imgs[:10]  # Fast mock check
        
    rgb_test_imgs = [p for p in all_test_imgs if p.name.startswith("rgb_")]
    thermal_test_imgs = [p for p in all_test_imgs if p.name.startswith("thermal_")]
    
    print(f"\n[INFO] Test Dataset Breakdown:")
    print(f"       Total test images   : {len(all_test_imgs)}")
    print(f"       RGB test images     : {len(rgb_test_imgs)}")
    print(f"       Thermal test images : {len(thermal_test_imgs)}")

    # 2. Write image lists and YAML files
    yaml_rgb = create_split_yaml(dataset_dir, rgb_test_imgs, "rgb")
    yaml_thermal = create_split_yaml(dataset_dir, thermal_test_imgs, "thermal")
    yaml_combined = create_split_yaml(dataset_dir, all_test_imgs, "combined")

    # Load YOLO model
    print(f"\n[INFO] Loading YOLO model: {model_path} ...")
    model = YOLO(str(model_path))

    # Evaluate RGB
    if len(rgb_test_imgs) > 0:
        print("\n" + "=" * 50)
        print("  Evaluating on RGB Test Set...")
        print("=" * 50)
        try:
            metrics_rgb = model.val(
                data=str(yaml_rgb),
                imgsz=parsed_args.imgsz if not parsed_args.mock else 320,
                conf=parsed_args.conf,
                iou=parsed_args.iou,
                workers=0,
                plots=True,
                save=True,
                verbose=False
            )
            print_performance_table("RGB Test Set", metrics_rgb)
            
            # Copy confusion matrix
            val_run_dir = Path(metrics_rgb.save_dir)
            if (val_run_dir / "confusion_matrix.png").exists():
                shutil.copy2(val_run_dir / "confusion_matrix.png", here / "confusion_matrix_rgb.png")
                shutil.copy2(val_run_dir / "confusion_matrix_normalized.png", here / "confusion_matrix_rgb_normalized.png")
                print(f"[OK] Saved RGB confusion matrices to: {here}")
        except Exception as e:
            print(f"[ERROR] RGB evaluation failed: {e}")
    else:
        print("[WARN] No RGB test images found. Skipping RGB evaluation.")

    # Evaluate Thermal
    if len(thermal_test_imgs) > 0:
        print("\n" + "=" * 50)
        print("  Evaluating on Thermal Test Set...")
        print("=" * 50)
        try:
            metrics_thermal = model.val(
                data=str(yaml_thermal),
                imgsz=parsed_args.imgsz if not parsed_args.mock else 320,
                conf=parsed_args.conf,
                iou=parsed_args.iou,
                workers=0,
                plots=True,
                save=True,
                verbose=False
            )
            print_performance_table("Thermal Test Set", metrics_thermal)
            
            # Copy confusion matrix
            val_run_dir = Path(metrics_thermal.save_dir)
            if (val_run_dir / "confusion_matrix.png").exists():
                shutil.copy2(val_run_dir / "confusion_matrix.png", here / "confusion_matrix_thermal.png")
                shutil.copy2(val_run_dir / "confusion_matrix_normalized.png", here / "confusion_matrix_thermal_normalized.png")
                print(f"[OK] Saved Thermal confusion matrices to: {here}")
        except Exception as e:
            print(f"[ERROR] Thermal evaluation failed: {e}")
    else:
        print("[WARN] No Thermal test images found. Skipping Thermal evaluation.")

    # Evaluate Combined
    print("\n" + "=" * 50)
    print("  Evaluating on Combined Test Set...")
    print("=" * 50)
    try:
        metrics_combined = model.val(
            data=str(yaml_combined),
            imgsz=parsed_args.imgsz if not parsed_args.mock else 320,
            conf=parsed_args.conf,
            iou=parsed_args.iou,
            workers=0,
            plots=True,
            save=True,
            verbose=False
        )
        print_performance_table("Combined Test Set", metrics_combined)
        
        # Copy confusion matrix
        val_run_dir = Path(metrics_combined.save_dir)
        if (val_run_dir / "confusion_matrix.png").exists():
            shutil.copy2(val_run_dir / "confusion_matrix.png", here / "confusion_matrix_combined.png")
            shutil.copy2(val_run_dir / "confusion_matrix_normalized.png", here / "confusion_matrix_combined_normalized.png")
            print(f"[OK] Saved Combined confusion matrices to: {here}")
    except Exception as e:
        print(f"[ERROR] Combined evaluation failed: {e}")

    # Clean up temp files
    for name in ["rgb", "thermal", "combined"]:
        for suffix in [".txt", ".yaml"]:
            fpath = dataset_dir / f"data_val_{name}{suffix}" if suffix == ".yaml" else dataset_dir / f"test_{name}{suffix}"
            if fpath.exists():
                try:
                    fpath.unlink()
                except Exception:
                    pass

    print("\n" + "=" * 60)
    print("  🎉  EVALUATION PROCESS COMPLETE!")
    print("=" * 60)
    print(f"   RGB Confusion Matrix      : {here / 'confusion_matrix_rgb.png'}")
    print(f"   Thermal Confusion Matrix  : {here / 'confusion_matrix_thermal.png'}")
    print(f"   Combined Confusion Matrix : {here / 'confusion_matrix_combined.png'}")
    print()

if __name__ == "__main__":
    main()
