from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

import cv2
import pandas as pd

from src.data.dataset import DatasetPaths

@dataclass(frozen=True)
class VideoValidationResult:
    """Validation information collected for one video"""
    video: str
    path: Path | None
    exists: bool
    opened: bool
    metadata_frames: int
    actual_frames: int
    fps: float
    width: int
    height: int
    frame_count_matches: bool
    frames_decoded: bool
    max_annotated_frame: int | None
    annotation_within_video: bool

def find_video_file(videos_dir: Path, video_name: str) -> Path | None:
    """Find a video file whose filename matches the dataset video name."""
    matching_files = [
        path
        for path in videos_dir.iterdir()
        if path.is_file()
        and path.stem.casefold() == video_name.casefold()
    ]
    if not matching_files:
        return None

    if len(matching_files) > 1:
        raise ValueError(
            f"Multiple video files found for '{video_name}': "
            f"{matching_files}"
        )

    return matching_files[0]

# def _check_frame_decoding(
#     capture: cv2.VideoCapture,
#     frame_count: int,
# ) -> bool:
#     """Check whether representative frames can be decoded."""

#     if frame_count <= 0:
#         return False

#     positions = {
#         0,
#         frame_count // 2,
#         frame_count - 1,
#     }

#     for position in positions:
#         capture.set(cv2.CAP_PROP_POS_FRAMES, position)

#         success, frame = capture.read()

#         if not success or frame is None:
#             return False

#     return True

def _check_frame_decoding(
    capture: cv2.VideoCapture,
    frame_count: int,
    max_annotated_frame: int | None,
) -> bool:
    """Check whether representative dataset-relevant frames can be decoded."""

    if frame_count <= 0:
        return False

    positions = [
        0,
        frame_count // 2,
    ]

    if max_annotated_frame is not None:
        annotated_position = max_annotated_frame - 1

        # Avoid testing the final physical frame because some videos
        # report it but OpenCV cannot reliably decode it.
        safe_annotated_position = min(
            annotated_position,
            frame_count - 2,
        )

        if safe_annotated_position >= 0:
            positions.append(safe_annotated_position)

    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)

        success, frame = capture.read()

        if not success or frame is None:
            return False

    return True

def validate_video(
    video_path: Path,
    expected_frames: int,
    max_annotated_frame: int | None,
) -> dict:
    """Validate one video file using OpenCV."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return {
            "opened": False,
            "actual_frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "frame_count_matches": False,
            "frames_decoded": False,
        }

    actual_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    frames_decoded = _check_frame_decoding(
        capture,
        actual_frames,
        max_annotated_frame,
    )

    print(
    f"Validating video: {video_path.name} | "
    f"actual_frames={actual_frames} | "
    f"max_annotated_frame={max_annotated_frame}"
    )

    annotation_within_video = (
        max_annotated_frame is not None
        and max_annotated_frame <= actual_frames
    )

    capture.release()

    return {
        "opened": True,
        "actual_frames": actual_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count_matches": actual_frames == expected_frames,
        "frames_decoded": frames_decoded,
        "max_annotated_frame": max_annotated_frame,
        "annotation_within_video": annotation_within_video,
    }

def validate_all_videos(
    paths: DatasetPaths,
    metadata_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Validate every video referenced by the dataset metadata."""

    max_annotated_frames = (
        annotations_df.groupby("video")["end_frame"].max()
    )
    results = []

    for _, row in tqdm(metadata_df.iterrows(),
                       total=len(metadata_df),
                       desc="Validating videos"):

        video_name = str(row["Video Name"])
        expected_frames = int(row["Frames"])
        max_annotated_frame = max_annotated_frames.get(video_name)

        video_path = find_video_file(
            paths.videos,
            video_name,
        )

        if video_path is None:
            results.append(
                VideoValidationResult(
                    video=video_name,
                    path=None,
                    exists=False,
                    opened=False,
                    metadata_frames=expected_frames,
                    actual_frames=0,
                    fps=0.0,
                    width=0,
                    height=0,
                    frame_count_matches=False,
                    frames_decoded=False,
                    max_annotated_frame=max_annotated_frame,
                    annotation_within_video=False
                ).__dict__
            )

            continue

        validation = validate_video(
            video_path,
            expected_frames,
            max_annotated_frame,
        )

        results.append(
            VideoValidationResult(
                video=video_name,
                path=video_path,
                exists=True,
                **validation,
                metadata_frames=expected_frames,
            ).__dict__
        )

    return pd.DataFrame(results)