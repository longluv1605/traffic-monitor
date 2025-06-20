import os
import yaml
from dotenv import load_dotenv
from roboflow import Roboflow


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_datasets(config: dict) -> None:
    """Fine-tune YOLOv8 model on custom dataset.

    Args:
        config: Dictionary containing fine-tune parameters.

    Returns:
        Trained YOLO model.
    """
    load_dotenv()
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
    MODEL_FORMAT = config['roboflow']['model_format']
    WORKSPACE = config['roboflow']['workspace']
    PROJECT = config['roboflow']['project']
    VERSION = config['roboflow']['version']
    LOCATION = config['roboflow']['location']
    
    print("Starting download dataset...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY, model_format=MODEL_FORMAT)
    dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(location=LOCATION)

def main():
    """Main function to fine-tune and evaluate YOLOv8 model."""
    config = load_config()
    load_datasets(config=config)


if __name__ == "__main__":
    main()