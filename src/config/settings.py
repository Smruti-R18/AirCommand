from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")

def load_config(config_path=CONFIG_PATH):
    """Load configuration from a YAML file."""
    with config_path.open("r",encoding="utf-8") as file:
        return yaml.safe_load(file)

CONFIG = load_config()

DATASET_ROOT = Path(
    os.environ["AIRCOMMAND_DATASET_ROOT"]
).expanduser().resolve()

MEDIAPIPE_HAND_LANDMARKER_MODEL = (
    PROJECT_ROOT
    / CONFIG["models"]["mediapipe_hand_landmarker"]
)

def validate_dataset_root(dataset_root):
    """Validate that the configured dataset directory exists."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Configured dataset path is not a directory: {dataset_root}")

def validate_file_exists(file_path: Path, name: str) -> None:
    """Validate that a required file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"{name} does not exist: {file_path}"
        )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"{name} is not a file: {file_path}"
        )

validate_file_exists(
    MEDIAPIPE_HAND_LANDMARKER_MODEL,
    "MediaPipe hand landmarker model",
)