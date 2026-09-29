from dataclasses import dataclass
from pathlib import Path
from src.config.settings import DATASET_ROOT

@dataclass(frozen=True)
class DatasetPaths:
    """Paths to the resources required by the IPN Hand dataset."""
    root: Path
    videos: Path
    annotations: Path
    metadata: Path
    class_details: Path
    train_annotations: Path
    test_annotations: Path

def discover_dataset_paths(root: Path = DATASET_ROOT) -> DatasetPaths:
    """Discover IPN hand dataset resources."""
    annotations_root = root/"annotations"/"annotations"
    videos_root = root/"videos"/"videos"
    return DatasetPaths(
        root=root,
        videos=videos_root,
        annotations=annotations_root/"Annot_List.txt",
        metadata=annotations_root/"metadata.xlsx",
        class_details=annotations_root/"class_details.txt",
        train_annotations=annotations_root/"Annot_TrainList.txt", 
        test_annotations=annotations_root/"Annot_TestList.txt"    
    )

def validate_dataset_paths(paths: DatasetPaths):
    """Validate that required dataset resources exist."""
    required_paths = {
        "dataset_root": paths.root,
        "videos directory": paths.videos,
        "annotations": paths.annotations,
        "metadata": paths.metadata,
        "class details": paths.class_details,
        "train annotations": paths.train_annotations,
        "test annotations": paths.test_annotations
    }

    missing_paths = [
        f"{name} : {path}"
        for name,path in required_paths.items()
        if not path.exists()
    ]

    if missing_paths:
        details = "\n".join(missing_paths)
        raise FileNotFoundError(
            f"The following required dataset resources are missing:\n{details}"
        )

def get_dataset_paths() -> DatasetPaths:
    """Discover and validate the IPN Hand dataset"""
    paths = discover_dataset_paths()
    validate_dataset_paths(paths)
    return paths