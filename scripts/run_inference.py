# run_inference.py
"""
Real-time / Static Bounding Box Inference Tool for Multiclass Drone Detection.
Supports webcams (integers), video files, and images.
"""

import sys
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

# Class colors: BGR format
CLASS_COLORS = {
    0: (0, 0, 255),     # Drone -> Bright Red
    1: (0, 255, 0),     # Bird -> Neon Green
    2: (255, 0, 0),     # Plane -> Deep Blue
    3: (0, 165, 255)    # Kite -> Vibrant Orange
}

CLASS_NAMES = {
    0: "Drone",
    1: "Bird",
    2: "Plane",
    3: "Kite"
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

def find_latest_model() -> str:
    """Finds the best.pt weights in runs/detect/."""
    here = Path(__file__).resolve().parent
    search_root = here / "runs" / "detect"
    candidates = list(search_root.rglob("best.pt"))
    if not candidates:
        print("[ERROR] No trained best.pt model weights found in runs/detect/.")
        print("        Please run training or provide the path with the --weights flag.")
        sys.exit(1)
    # Get latest modified file
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())

def draw_detections(frame, results):
    """Draws custom colored bounding boxes and labels on the frame."""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # Class index and confidence
            cls_idx = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # Define color and label
            color = CLASS_COLORS.get(cls_idx, (255, 255, 255))
            cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
            label = f"{cls_name} {conf:.2f}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0], y1), color, -1)
            
            # Put text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
    return frame

def main():
    parser = argparse.ArgumentParser(description="Run inference on images, videos, or webcams.")
    parser.add_argument("--source", type=str, required=True, help="Path to image/video file, or integer index (e.g. 0) for webcam.")
    parser.add_argument("--weights", type=str, default=None, help="Path to best.pt weights (auto-detects latest run if omitted).")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    parser.add_argument("--save", action="store_true", help="Save the annotated output file.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI windows displayed).")
    args = parser.parse_args()

    # Determine weights path
    weights_path = args.weights if args.weights else find_latest_model()
    print(f"[INFO] Loading model weights: {weights_path}")
    model = YOLO(weights_path)

    # Parse source
    source = args.source
    is_webcam = False
    try:
        # Check if source is an integer (e.g. "0")
        source = int(source)
        is_webcam = True
    except ValueError:
        pass

    # Check if the source is a static image
    is_image = False
    if not is_webcam:
        path = Path(source)
        if not path.exists():
            print(f"[ERROR] Source path does not exist: {source}")
            sys.exit(1)
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            is_image = True

    if is_image:
        print(f"[INFO] Running inference on static image: {source}")
        frame = cv2.imread(str(source))
        if frame is None:
            print("[ERROR] Failed to load image.")
            sys.exit(1)
            
        results = model.predict(frame, conf=args.conf, verbose=False)
        annotated_frame = draw_detections(frame, results)
        
        # Save output if requested
        if args.save:
            out_path = Path("inference_output_" + Path(source).name)
            cv2.imwrite(str(out_path), annotated_frame)
            print(f"[OK] Saved annotated image to: {out_path.resolve()}")
            
        if not args.headless:
            # Display image
            cv2.imshow("Multiclass Drone Detection - Image Inference", annotated_frame)
            print("[INFO] Press any key in the image window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    else:
        # Webcam or Video
        print(f"[INFO] Initializing video capture: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video source: {source}")
            sys.exit(1)
            
        # Get properties for saving
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        writer = None
        if args.save and not is_webcam:
            out_path = Path("inference_output_" + Path(args.source).stem + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            print(f"[INFO] Saving video output to: {out_path.resolve()}")

        if not args.headless:
            print("[INFO] Inference active. Press 'q' in the window to terminate.")
            
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame, conf=args.conf, verbose=False)
            annotated_frame = draw_detections(frame, results)
            
            # Show live window if not headless
            if not args.headless:
                cv2.imshow("Multiclass Drone Detection - Video Inference", annotated_frame)
                # Exit check
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if writer:
                writer.write(annotated_frame)
            
            frame_idx += 1
            if args.headless and frame_idx % 30 == 0:
                print(f"[INFO] Processed {frame_idx} frames...")
                
        cap.release()
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Inference process terminated.")

if __name__ == "__main__":
    main()
