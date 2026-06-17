# generate_pseudo_labels.py
"""
Generates pseudo-labels from a test video (Video Project 2.mp4).
Uses the trained model to find bounding boxes, but forces the class to 0 (Drone).
Saves the results to a folder structure compatible with prepare_multiclass_dataset.py.
"""

import os
import cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    model_path = Path(r"D:\DRONE\New folder (2)\runs\detect\consolidated_transfer_yolo11\weights\best.pt")
    video_path = Path(r"D:\DRONE\New folder (2)\Video Project 2.mp4")
    output_dir = Path(r"D:\DRONE\New folder (2)\Video_Project_2_override")

    print("=" * 60)
    print("  [PSEUDO-LABEL] GENERATING DRONE CLASS OVERRIDES")
    print("=" * 60)
    print(f"Model Path : {model_path}")
    print(f"Video Path : {video_path}")
    print(f"Output Dir : {output_dir}")

    if not model_path.exists():
        print(f"[ERROR] Model weights not found at: {model_path}")
        return
    if not video_path.exists():
        print(f"[ERROR] Video file not found at: {video_path}")
        return

    # Create directory structure
    img_dir = output_dir / "train" / "images"
    lbl_dir = output_dir / "train" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    
    frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference to detect where the object is
        results = model.predict(frame, conf=0.20, verbose=False)[0]
        
        if len(results.boxes) > 0:
            # We found the drone! Create the annotation
            new_lines = []
            for box in results.boxes:
                # Force the class ID to 0 (Drone)
                class_id = 0
                
                # Get normalized box coordinates
                # xywhn returns [x_center, y_center, width, height] normalized
                xywhn = box.xywhn[0].cpu().numpy()
                x_center, y_center, w, h = xywhn
                
                new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # Define filenames
            img_name = f"vp2_frame_{frame_idx:05d}.jpg"
            lbl_name = f"vp2_frame_{frame_idx:05d}.txt"

            # Save frame and label file
            cv2.imwrite(str(img_dir / img_name), frame)
            (lbl_dir / lbl_name).write_text("\n".join(new_lines) + "\n")
            
            extracted_count += 1

        frame_idx += 1

    cap.release()
    print("\n" + "=" * 60)
    print("  [OK] PSEUDO-LABEL GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Total Frames Processed : {frame_idx}")
    print(f"Total Labeled Frames   : {extracted_count}")
    print(f"Output saved to        : {output_dir.resolve()}")

if __name__ == "__main__":
    main()
