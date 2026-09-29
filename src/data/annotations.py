from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.dataset import DatasetPaths


@dataclass(frozen=True)
class AnnotationRecord:
    """One annotated segment from an IPN Hand video."""
    video: str
    label: str
    class_id: int
    start_frame: int
    end_frame: int
    frame_count: int


@dataclass(frozen=True)
class ClassInfo:
    """Information describing one IPN Hand class."""
    class_id: int
    label: str
    gesture: str


def load_annotations(annotation_path: Path) -> pd.DataFrame:
    """
    Load an IPN Hand annotation file into a DataFrame.
    """
    column_names = [
            "video",
            "label",
            "id",
            "t_start",
            "t_end",
            "frames",
    ]
    dataframe = pd.read_csv(annotation_path,header=None,names=column_names)
    return dataframe.rename(
        columns={
            "id":"class_id",
            "t_start":"start_frame",
            "t_end":"end_frame",
            "frames":"frame_count"
        }
    )

def load_class_details(class_details_path: Path) -> pd.DataFrame:
    """
    Load IPN Hand class definitions.
    """

    dataframe = pd.read_csv(
        class_details_path,
        sep="\t",
        header=0,
        usecols=["id", "Label", "Gesture"],
    )

    dataframe = dataframe.rename(
        columns={
            "id": "class_id",
            "Label": "label",
            "Gesture": "gesture",
        }
    )

    return dataframe

def load_train_annotations(paths: DatasetPaths) -> pd.DataFrame:
    """Load annotations belonging to the official training split."""

    return load_annotations(paths.train_annotations)


def load_test_annotations(paths: DatasetPaths) -> pd.DataFrame:
    """Load annotations belonging to the official test split."""

    return load_annotations(paths.test_annotations)


def load_class_mapping(paths: DatasetPaths) -> pd.DataFrame:
    """Load the mapping between class IDs, labels and gesture names."""

    return load_class_details(paths.class_details)