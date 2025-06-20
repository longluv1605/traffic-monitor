import cv2
import numpy as np
import pandas as pd
import time
import random
import yaml
from collections import deque
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def generate_colors(num_colors: int) -> List[Tuple[int, int, int]]:
    """Generate a list of distinct RGB colors.

    Args:
        num_colors: Number of colors to generate.

    Returns:
        List of RGB tuples (0-255).
    """
    random.seed(42)  # For reproducibility
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        # Avoid very bright or dark colors
        if r + g + b < 100 or r + g + b > 600:
            r = random.randint(100, 200)
            g = random.randint(100, 200)
            b = random.randint(100, 200)
        colors.append((r, g, b))
    return colors


def load_model(model_path: str) -> Tuple[YOLO, Dict[int, Tuple[int, int, int]]]:
    """Load YOLOv8 model and generate class colors.

    Args:
        model_path: Path to model weights (.pt).

    Returns:
        Tuple of (loaded YOLO model, dictionary of class colors).
    """
    print(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    class_names = model.names  # Get class names from model
    colors = generate_colors(len(class_names))
    class_colors = {i: color for i, color in enumerate(colors)}
    return model, class_colors


def run_object_detection(model: YOLO, frame: np.ndarray, device: str, class_colors: Dict[int, Tuple[int, int, int]],
                         conf_threshold: float, imgsz: int) -> Tuple[List, float, float, Dict[str, int]]:
    """Perform object detection on a frame.

    Args:
        model: YOLOv8 model.
        frame: Input frame (BGR).
        class_colors: Dictionary mapping class IDs to RGB colors.
        conf_threshold: Confidence threshold for detections.
        imgsz: Image size for inference.

    Returns:
        Tuple of (detections, fps, latency, class_counts).
    """
    start_time = time.time()
    detections = []
    class_counts = {model.names[i]: 0 for i in range(len(model.names))}

    results = model(frame, device=device)
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for box, conf, cls in zip(boxes, confidences, classes):
            cls = int(cls)
            if conf > conf_threshold and cls in class_colors:
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
                class_counts[model.names[cls]] += 1

    latency = time.time() - start_time
    fps = 1 / latency if latency > 0 else float('inf')
    return detections, fps, latency, class_counts


def run_object_tracking(tracker: DeepSort, detections: List,
                        frame: np.ndarray) -> List[Tuple[int, List[float]]]:
    """Track objects using DeepSORT.

    Args:
        tracker: DeepSORT tracker instance.
        detections: List of detections from object detection.
        frame: Input frame (BGR).

    Returns:
        List of tracked objects (track_id, ltrb).
    """
    tracks = tracker.update_tracks(detections, frame=frame)
    tracked_objects = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        tracked_objects.append((track_id, ltrb, track.det_class))  # Include class
    return tracked_objects


def analyze_behavior(trajectories: Dict[int, deque], track_id: int, center_x: float,
                     center_y: float, frame_idx: int, wrong_way_threshold: int) -> List[Tuple[int, str, int]]:
    """Analyze vehicle behavior based on trajectory.

    Args:
        trajectories: Dictionary of vehicle trajectories.
        track_id: ID of the tracked vehicle.
        center_x: X-coordinate of vehicle center.
        center_y: Y-coordinate of vehicle center.
        frame_idx: Current frame index.
        wrong_way_threshold: Number of frames to detect wrong way.

    Returns:
        List of detected behaviors (track_id, behavior, frame_idx).
    """
    behaviors = []
    if track_id not in trajectories:
        trajectories[track_id] = deque(maxlen=50)
    trajectories[track_id].append((center_x, center_y, frame_idx))

    if len(trajectories[track_id]) >= wrong_way_threshold:
        # Detect wrong way (assume upward is correct, downward is wrong)
        y_positions = [pos[1] for pos in list(trajectories[track_id])[-wrong_way_threshold:]]
        delta_y = y_positions[-1] - y_positions[0]
        if delta_y < -50:  # Moved downward > 50 pixels
            behaviors.append((track_id, "Wrong Way", frame_idx))
            print((track_id, "Wrong Way", frame_idx))

    return behaviors


def evaluate_tracking_metrics(tracked_objects: List[Tuple[int, List[float]]],
                             ground_truth_tracks: List) -> Dict[str, float]:
    """Evaluate tracking metrics.

    Args:
        tracked_objects: List of tracked objects.
        ground_truth_tracks: List of ground truth tracks.

    Returns:
        Dictionary of tracking metrics.
    """
    matches = 0
    mismatches = 0
    misses = len(ground_truth_tracks) if ground_truth_tracks else 0
    for gt in ground_truth_tracks or []:
        for track in tracked_objects:
            if gt[0] == track[0]:
                matches += 1
                misses -= 1
                break
        else:
            mismatches += 1
    mota = (matches - mismatches - misses) / max(1, len(ground_truth_tracks)) if ground_truth_tracks else 0
    return {"MOTA": mota}


def evaluate_behavior_metrics(behaviors: List[Tuple[int, str, int]],
                             ground_truth_behaviors: List) -> Dict[str, float]:
    """Evaluate behavior analysis metrics.

    Args:
        behaviors: List of predicted behaviors.
        ground_truth_behaviors: List of ground truth behaviors.

    Returns:
        Dictionary of behavior metrics.
    """
    pred_labels = [b[1] for b in behaviors]
    true_labels = [b[1] for b in ground_truth_behaviors
                   if b[0] in [x[0] for x in behaviors]] if ground_truth_behaviors else []
    if not true_labels or not pred_labels:
        return {"Accuracy": 0, "FPR": 0}
    precision, recall, _, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted", zero_division=0
    )
    fpr = 1 - precision
    return {"Accuracy": (precision * recall) / max(precision + recall, 1e-6), "FPR": fpr}


def main():
    """Main function to run the traffic monitoring pipeline."""
    # Load configuration
    config = load_config()
    traffic_config = config['traffic_monitor']

    # 1. Load model and class colors
    model, class_colors = load_model(traffic_config['model_path'])

    # 2. Initialize DeepSORT
    tracker = DeepSort(max_age=30, nn_budget=100)

    # 3. Initialize video
    cap = cv2.VideoCapture(traffic_config['video_path'])
    trajectories = {}
    behaviors = []
    frame_idx = 0
    fps_list = []
    latency_list = []
    class_counts_list = []

    # 4. Open log file
    class_names = model.names
    log_header = "Frame,FPS,Latency_ms," + ",".join(class_names.values())
    with open(traffic_config['log_file'], "w") as f:
        f.write(f"{log_header}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 5. Object detection
        detections, fps, latency, class_counts = run_object_detection(
            model, frame,
            traffic_config['device'],
            class_colors,
            traffic_config['conf_threshold'], traffic_config['imgsz']
        )
        fps_list.append(fps)
        latency_list.append(latency)
        class_counts_list.append(class_counts)

        # 6. Object tracking
        tracked_objects = run_object_tracking(tracker, detections, frame)

        # 7. Behavior analysis
        for track_id, ltrb, cls in tracked_objects:
            center_x = (ltrb[0] + ltrb[2]) / 2
            center_y = (ltrb[1] + ltrb[3]) / 2
            new_behaviors = analyze_behavior(
                trajectories, track_id, center_x, center_y, frame_idx,
                traffic_config['wrong_way_threshold']
            )
            behaviors.extend(new_behaviors)

            # Draw results
            color = class_colors.get(cls, (255, 255, 255))  # Default white if not found
            cv2.rectangle(
                frame,
                (int(ltrb[0]), int(ltrb[1])),
                (int(ltrb[2]), int(ltrb[3])),
                color,
                2
            )
            cv2.putText(
                frame,
                f"{class_names[cls]}: {track_id}",
                (int(ltrb[0]), int(ltrb[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        # Display class counts and FPS
        y_offset = 60
        for class_name, count in class_counts.items():
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )
            y_offset += 20
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )
        # import matplotlib.pyplot as plt
        # plt.imshow(frame)
        # plt.show()
        
        cv2.imshow("Traffic Monitoring", frame)

        # Write log
        log_values = [str(class_counts.get(name, 0)) for name in class_names.values()]
        with open(traffic_config['log_file'], "a") as f:
            f.write(f"{frame_idx},{fps:.2f},{latency*1000:.2f},{','.join(log_values)}\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # 8. Save report
    behavior_df = pd.DataFrame(behaviors, columns=["Track ID", "Behavior", "Frame"])
    behavior_df.to_csv("behaviors.csv", index=False)
    print(
        f"Average FPS: {np.mean(fps_list):.2f}, "
        f"Average Latency: {np.mean(latency_list)*1000:.2f} ms"
    )
    for class_name in class_names.values():
        avg_count = np.mean([counts[class_name] for counts in class_counts_list])
        print(f"Average {class_name}: {avg_count:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()