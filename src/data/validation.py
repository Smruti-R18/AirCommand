from pathlib import Path

import pandas as pd

from src.data.dataset import DatasetPaths


REQUIRED_ANNOTATION_COLUMNS = {
    "video",
    "label",
    "class_id",
    "start_frame",
    "end_frame",
    "frame_count",
}


def validate_annotation_columns(dataframe: pd.DataFrame) -> None:
    """
    Validate that an annotation DataFrame contains the required columns.
    """

    missing_columns = REQUIRED_ANNOTATION_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing annotation columns: {sorted(missing_columns)}"
        )


def validate_frame_ranges(dataframe: pd.DataFrame) -> None:
    """
    Validate annotation frame ranges and frame counts.
    """

    invalid_ranges = dataframe[
        dataframe["start_frame"] > dataframe["end_frame"]
    ]

    if not invalid_ranges.empty:
        raise ValueError(
            f"Found {len(invalid_ranges)} annotations "
            "where start_frame is greater than end_frame."
        )

    expected_frame_count = (
        dataframe["end_frame"] - dataframe["start_frame"] + 1
    )

    invalid_counts = dataframe[
        dataframe["frame_count"] != expected_frame_count
    ]

    if not invalid_counts.empty:
        raise ValueError(
            f"Found {len(invalid_counts)} annotations "
            "with inconsistent frame counts."
        )


def validate_video_references(
    dataframe: pd.DataFrame,
    videos_directory: Path,
) -> None:
    """
    Validate that every annotated video exists in the dataset.
    """

    video_files = {
        path.stem
        for path in videos_directory.iterdir()
        if path.is_file()
    }

    annotated_videos = set(dataframe["video"])

    missing_videos = annotated_videos - video_files

    if missing_videos:
        examples = sorted(missing_videos)[:10]

        raise FileNotFoundError(
            f"Found {len(missing_videos)} annotated videos "
            "that do not exist in the videos directory. "
            f"Examples: {examples}"
        )


def validate_train_test_split(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
) -> None:
    """
    Validate that no video appears in both train and test annotations.
    """

    train_videos = set(train_dataframe["video"])
    test_videos = set(test_dataframe["video"])

    overlapping_videos = train_videos & test_videos

    if overlapping_videos:
        examples = sorted(overlapping_videos)[:10]

        raise ValueError(
            f"Found {len(overlapping_videos)} videos appearing "
            "in both train and test splits. "
            f"Examples: {examples}"
        )


def validate_labels(
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    classes_dataframe: pd.DataFrame,
) -> None:
    """
    Validate that all annotation labels exist in the class mapping.
    """

    known_labels = set(classes_dataframe["label"])

    train_labels = set(train_dataframe["label"])
    test_labels = set(test_dataframe["label"])

    unknown_train_labels = train_labels - known_labels
    unknown_test_labels = test_labels - known_labels

    if unknown_train_labels:
        raise ValueError(
            f"Unknown labels in training data: "
            f"{sorted(unknown_train_labels)}"
        )

    if unknown_test_labels:
        raise ValueError(
            f"Unknown labels in test data: "
            f"{sorted(unknown_test_labels)}"
        )

def validate_dataset(
    paths: DatasetPaths,
    train_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    classes_dataframe: pd.DataFrame,
) -> None:
    """
    Run all basic dataset validation checks.
    """

    validate_annotation_columns(train_dataframe)
    validate_annotation_columns(test_dataframe)

    validate_frame_ranges(train_dataframe)
    validate_frame_ranges(test_dataframe)

    validate_video_references(
        train_dataframe,
        paths.videos,
    )

    validate_video_references(
        test_dataframe,
        paths.videos,
    )

    validate_train_test_split(
        train_dataframe,
        test_dataframe,
    )

    validate_labels(
        train_dataframe,
        test_dataframe,
        classes_dataframe,
    )