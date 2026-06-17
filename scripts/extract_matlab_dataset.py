# extract_matlab_dataset.py
"""
Extracts frames and YOLO annotations from the Matlab-based Drone-detection-dataset.
Decodes MCOS .mat ground truth files using the mcos-decoder library.
"""

import os
import cv2
from pathlib import Path
from mcos_decoder import load_groundtruth

def main():
    dataset_dir = Path(r"d:\DRONE\New folder (2)\Drone-detection-dataset-master")
    output_dir = Path(r"d:\DRONE\New folder (2)\Drone-detection-dataset-master-yolo")
    stride = 15  # Extract every 15th frame to avoid redundancy and keep size manageable

    print("=" * 60)
    print("  [EXTRACT] MATLAB DRONE DETECTION DATASET FRAME EXTRACTOR")
    print("=" * 60)
    print(f"Source Directory: {dataset_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Sampling Stride : {stride}")

    if not dataset_dir.exists():
        print(f"[ERROR] Source directory not found: {dataset_dir}")
        return

    # Create directories
    for modality in ["visible", "ir"]:
        (output_dir / modality / "train" / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / modality / "train" / "labels").mkdir(parents=True, exist_ok=True)

    # Walk through folders
    video_folders = {
        "visible": dataset_dir / "Data" / "Video_V",
        "ir": dataset_dir / "Data" / "Video_IR"
    }

    total_extracted = 0

    for modality, folder in video_folders.items():
        if not folder.exists():
            print(f"[WARN] Folder not found: {folder}")
            continue

        print(f"\nProcessing {modality} videos...")
        video_paths = sorted(list(folder.glob("*.mp4")))
        print(f"Found {len(video_paths)} videos in {folder.name}")

        for v_idx, v_path in enumerate(video_paths):
            video_name = v_path.stem
            
            # Map video category based on name
            upper_name = video_name.upper()
            if "DRONE" in upper_name:
                class_id = 0
            elif "BIRD" in upper_name:
                class_id = 1
            elif "AIRPLANE" in upper_name:
                class_id = 2
            elif "HELICOPTER" in upper_name:
                class_id = 2
            else:
                print(f"   [SKIP] Unknown class for video: {video_name}")
                continue

            # Find corresponding label .mat file
            label_path = folder / f"{video_name}_LABELS.mat"
            if not label_path.exists():
                print(f"   [WARN] Labels not found for: {video_name} (expected {label_path.name})")
                continue

            try:
                bboxes = load_groundtruth(str(label_path))
            except Exception as e:
                print(f"   [ERROR] Failed to decode labels for {video_name}: {e}")
                continue

            # Open video
            cap = cv2.VideoCapture(str(v_path))
            if not cap.isOpened():
                print(f"   [ERROR] Failed to open video: {v_path}")
                continue

            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            frame_idx = 0
            extracted_from_video = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check stride
                if frame_idx % stride == 0:
                    # Get bbox if within bounds
                    bbox = None
                    if frame_idx < len(bboxes):
                        bbox = bboxes[frame_idx]

                    new_lines = []
                    if bbox is not None:
                        x, y, w, h = bbox
                        # Convert Matlab [x_min, y_min, w, h] to YOLO normalized [x_center, y_center, w, h]
                        x_center = (x + w / 2.0) / width
                        y_center = (y + h / 2.0) / height
                        norm_w = w / width
                        norm_h = h / height
                        
                        # Clip to [0, 1] range to avoid floating-point noise out of bounds
                        x_center = max(0.0, min(1.0, x_center))
                        y_center = max(0.0, min(1.0, y_center))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))
                        
                        new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

                    # Define save paths
                    img_save_name = f"{video_name}_frame_{frame_idx:05d}.jpg"
                    lbl_save_name = f"{video_name}_frame_{frame_idx:05d}.txt"
                    
                    img_save_path = output_dir / modality / "train" / "images" / img_save_name
                    lbl_save_path = output_dir / modality / "train" / "labels" / lbl_save_name
                    
                    # Write image and label
                    cv2.imwrite(str(img_save_path), frame)
                    lbl_save_path.write_text("\n".join(new_lines) + "\n" if new_lines else "")
                    
                    extracted_from_video += 1
                    total_extracted += 1

                frame_idx += 1

            cap.release()
            
            if (v_idx + 1) % 50 == 0 or v_idx == len(video_paths) - 1:
                print(f"   Processed {v_idx + 1}/{len(video_paths)} videos (Extracted {extracted_from_video} frames from this video)")

    print("\n" + "=" * 60)
    print("  [OK] EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"Total Extracted YOLO Frames: {total_extracted}")
    print(f"Destination folder: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
