# train_transfer.py
"""
Fine-tunes YOLOv11x on the multiclass drone dataset (Drone, Bird, Plane, Kite)
using transfer learning. Freezes the backbone layers (0-10) to preserve 
pre-trained weights and trains the neck and head layers.
"""

import sys
import os
import argparse
import time
import torch
import yaml
from pathlib import Path
from ultralytics import YOLO

def print_banner():
    print("\n" + "=" * 60)
    print("  [TRAIN] YOLOv11 MULTICLASS TRANSFER LEARNING PIPELINE")
    print("=" * 60)

def check_device():
    if torch.cuda.is_available():
        device = 0
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n[INFO] GPU detected : {name}")
        print(f"       VRAM         : {vram:.1f} GB")
    else:
        device = "cpu"
        print(f"\n[WARN] No GPU detected. Training on CPU. This will be very slow.")
    return device

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Transfer Learning for Multiclass Drone Detection.")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Path to base weights (default: yolo11x.pt).")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs (default: 80).")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto-config based on VRAM).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640).")
    parser.add_argument("--mock", action="store_true", help="Run a quick 1-epoch sanity check pass on 2 batches.")
    parsed_args = parser.parse_args()

    device = check_device()
    here = Path(__file__).resolve().parent
    
    # Locate data_multiclass.yaml
    yaml_path = here / "merged_multiclass_dataset" / "data_multiclass.yaml"
    if not yaml_path.exists():
        print(f"\n[ERROR] Dataset config not found at: {yaml_path}")
        print("        Did you run prepare_multiclass_dataset.py first?")
        sys.exit(1)
    
    print(f"\n[INFO] Using dataset config: {yaml_path.resolve()}")

    # Determine hyperparameters based on args
    epochs = parsed_args.epochs
    img_size = parsed_args.imgsz
    
    # Auto-batch config based on GPU memory if batch is -1
    if parsed_args.batch != -1:
        batch = parsed_args.batch
    else:
        if device == 0:
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram >= 16:
                batch = 16
            elif vram >= 8:
                batch = 8
            elif vram >= 6:
                batch = 4
            else:
                batch = 2
        else:
            batch = 2
            
    # Mock settings override
    mock_yaml_path = None
    if parsed_args.mock:
        print("\n[INFO] --- RUNNING MOCK TEST PASS ---")
        epochs = 1
        batch = 2
        img_size = 320  # Fast resize for mock
        
        # Create a tiny mock dataset
        mock_yaml_path = here / "merged_multiclass_dataset" / "data_mock.yaml"
        train_mock_txt = here / "merged_multiclass_dataset" / "train_mock.txt"
        val_mock_txt = here / "merged_multiclass_dataset" / "val_mock.txt"
        
        # Select 8 images from train and 4 from val
        train_imgs = list((here / "merged_multiclass_dataset" / "train" / "images").glob("*.*"))[:8]
        val_imgs = list((here / "merged_multiclass_dataset" / "valid" / "images").glob("*.*"))[:4]
        
        # Write files
        with open(train_mock_txt, "w") as f:
            for p in train_imgs:
                f.write(str(p.resolve()) + "\n")
        with open(val_mock_txt, "w") as f:
            for p in val_imgs:
                f.write(str(p.resolve()) + "\n")
                
        mock_cfg = {
            "path": str((here / "merged_multiclass_dataset").resolve()),
            "train": "train_mock.txt",
            "val": "val_mock.txt",
            "nc": 4,
            "names": {
                0: "Drone",
                1: "Bird",
                2: "Plane",
                3: "Kite"
            }
        }
        with open(mock_yaml_path, "w") as f:
            yaml.dump(mock_cfg, f, default_flow_style=False, sort_keys=False)
            
        yaml_path = mock_yaml_path
        print(f"[INFO] Created temporary mock dataset config: {yaml_path}")

    print(f"\n[INFO] Training Parameters:")
    print(f"       Base Model : {parsed_args.model}")
    print(f"       Epochs     : {epochs}")
    print(f"       Batch Size : {batch}")
    print(f"       Image Size : {img_size}")
    print(f"       Device     : {device}")
    
    # Load model weights
    print(f"\n[INFO] Loading base weights: {parsed_args.model} ...")
    model = YOLO(parsed_args.model)
    
    # Print the model layer names and freeze status
    print("\n[INFO] Freezing backbone layers (0 to 10)...")
    # In YOLOv11: layers 0-10 are the backbone, 11-22 are neck, 23 is detection head.
    # Programmatic freezing for logging
    freeze_prefixes = [f"model.{i}." for i in range(11)]
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.model.named_parameters():
        is_frozen = any(p in name for p in freeze_prefixes)
        if is_frozen:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    print(f"       Frozen parameters    : {frozen_params:,} ({frozen_params / (frozen_params + trainable_params) * 100:.1f}%)")
    print(f"       Trainable parameters : {trainable_params:,} ({trainable_params / (frozen_params + trainable_params) * 100:.1f}%)")

    # Define runs folders
    run_project = str(here / "runs" / "detect")
    run_name = "multiclass_transfer_yolo11x"
    if parsed_args.mock:
        run_name += "_mock"

    print(f"\n[INFO] Training output project folder: {run_project}/{run_name}")
    print("\n[INFO] Starting training...")
    
    t0 = time.time()
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=img_size,
            batch=batch,
            project=run_project,
            name=run_name,
            exist_ok=True,
            patience=15 if not parsed_args.mock else 1,
            device=device,
            # Transfer learning backbone freeze
            freeze=11,  # Freezes modules 0 to 10 natively in Ultralytics
            # Optimizer & Tuning
            optimizer="AdamW",
            lr0=0.001,  # Lower LR for fine-tuning
            lrf=0.01,
            weight_decay=0.001,
            warmup_epochs=1 if parsed_args.mock else 3,
            # Bounding box / target enhancements (scale and small object tuning)
            augment=True,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.15,
            degrees=15.0,
            translate=0.15,
            scale=0.5,
            shear=2.0,
            fliplr=0.5,
            flipud=0.2,  # Drones/birds can be viewed at upside down angles
            label_smoothing=0.05,
            plots=True,
            save=True,
            workers=0,  # Recommended 0 on Windows to avoid process spawning lockups
            val=True
        )
        
        elapsed = time.time() - t0
        mins, secs = divmod(int(elapsed), 60)
        print("\n" + "=" * 60)
        print(f"  [OK] TRAINING COMPLETED IN {mins}m {secs}s!")
        print("=" * 60)
        
        # Output locations of models
        best_pt = Path(run_project) / run_name / "weights" / "best.pt"
        last_pt = Path(run_project) / run_name / "weights" / "last.pt"
        print(f"   Best Weights : {best_pt.resolve()} ({'Found' if best_pt.exists() else 'Not Found'})")
        print(f"   Last Weights : {last_pt.resolve()} ({'Found' if last_pt.exists() else 'Not Found'})")
        
    except torch.cuda.OutOfMemoryError:
        print("\n[ERROR] GPU Out Of Memory. Try reducing the batch size using the --batch argument.")
    except Exception as e:
        print(f"\n[ERROR] Training failed with error: {e}")
    finally:
        if parsed_args.mock and mock_yaml_path:
            for fpath in [train_mock_txt, val_mock_txt, mock_yaml_path]:
                if fpath.exists():
                    try:
                        fpath.unlink()
                    except Exception:
                        pass

if __name__ == "__main__":
    main()
