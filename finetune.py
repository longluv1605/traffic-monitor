import os
import yaml
from ultralytics import YOLO


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def fine_tune_yolov8(config: dict) -> YOLO:
    """Fine-tune YOLOv8 model on custom dataset.

    Args:
        config: Dictionary containing fine-tune parameters.

    Returns:
        Trained YOLO model.
    """
    print("Starting fine-tuning YOLOv8...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=config['finetune']['data_yaml'],
        epochs=config['finetune']['epochs'],
        imgsz=config['finetune']['imgsz'],
        batch=config['finetune']['batch_size'],
        device=config['finetune']['device'],
        patience=10,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        project="runs/train",
        name="traffic_monitoring",
        exist_ok=True
    )
    metrics = model.val()
    print(f"Fine-tuning metrics: mAP@0.5: {metrics.box.map50:.4f}, "
          f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    return model


def evaluate_detection_metrics(model: YOLO, val_data_yaml: str) -> dict:
    """Evaluate detection metrics on validation set.

    Args:
        model: YOLOv8 model.
        val_data_yaml: Path to validation dataset configuration.

    Returns:
        Dictionary of detection metrics.
    """
    metrics = model.val(data=val_data_yaml)
    return {
        "mAP@0.5": metrics.box.map50,
        "mAP@0.5:0.95": metrics.box.map,
        "Precision": metrics.box.p,
        "Recall": metrics.box.r
    }


def main():
    """Main function to fine-tune and evaluate YOLOv8 model."""
    config = load_config()
    model_path = config['traffic_monitor']['model_path']

    if not os.path.exists(model_path):
        model = fine_tune_yolov8(config)
    else:
        model = YOLO(model_path)

    detection_metrics = evaluate_detection_metrics(model, config['finetune']['data_yaml'])
    print("Detection Metrics:", detection_metrics)


if __name__ == "__main__":
    main()