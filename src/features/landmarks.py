from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


from src.config.settings import MEDIAPIPE_HAND_LANDMARKER_MODEL

@dataclass(frozen=True)
class LandmarkFrame:
    """Landmarks extracted from one video frame"""
    landmarks: np.ndarray
    hand_detected: bool

@dataclass(frozen=True)
class GestureSequence:
    """Processed landmark sequence for one annotated gesture segment."""
    video: str
    label: str
    class_id: int
    landmarks: np.ndarray
    hand_detected: np.ndarray

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_hand_detector():
    """Create and configure the MediaPipe hand landmarker."""

    base_options = BaseOptions(
        model_asset_path=str(MEDIAPIPE_HAND_LANDMARKER_MODEL)
    )

    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)

def create_gesture_sequence(
    video: str,
    label: str,
    class_id: int,
    landmarks: np.ndarray,
    hand_detected: np.ndarray,
) -> GestureSequence:
    """Create a processed gesture sequence"""
    return GestureSequence(
        video=video,
        label=label,
        class_id = class_id,
        landmarks = landmarks,
        hand_detected = hand_detected,
    )

def extract_landmarks(
    frame: np.ndarray,
    detector,
    timestamp_ms: int,
) -> LandmarkFrame:
    """Extract hand landmarks from one video frame."""

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms,
    )

    if not result.hand_landmarks:
        return LandmarkFrame(
            landmarks=np.zeros((21, 3), dtype=np.float32),
            hand_detected=False,
        )

    hand_landmarks = result.hand_landmarks[0]

    landmarks = np.array(
        [
            [landmark.x, landmark.y, landmark.z]
            for landmark in hand_landmarks
        ],
        dtype=np.float32,
    )

    return LandmarkFrame(
        landmarks=landmarks,
        hand_detected=True,
    )

def normalize_landmarks(landmarks: np.ndarray)-> np.ndarray:
    """Normalize hand landmarks relative to the wrist."""
    wrist = landmarks[:,0:1,:]
    normalized = landmarks - wrist
    return normalized

def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Flatten landmark coordinated while preserving the time dimension"""
    return landmarks.reshape(
        landmarks.shape[0],-1
    )

def extract_segment_landmarks(
    video_path,
    start_frame: int,
    end_frame: int,
    detector,
):
    """Extract hand landmarks from an annotated video segment."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Could not determine FPS for video: {video_path}"
        )

    landmarks_sequence = []
    detection_sequence = []

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    for frame_index in range(start_frame, end_frame + 1):

        success, frame = capture.read()

        if not success or frame is None:
            capture.release()
            raise RuntimeError(
                f"Could not decode frame {frame_index} "
                f"from video: {video_path}"
            )

        timestamp_ms = int(
            frame_index / fps * 1000
        )

        result = extract_landmarks(
            frame,
            detector,
            timestamp_ms,
        )

        landmarks_sequence.append(
            result.landmarks
        )

        detection_sequence.append(
            result.hand_detected
        )

    capture.release()

    landmarks_sequence = np.stack(
        landmarks_sequence,
        axis=0,
    )

    detection_sequence = np.array(
        detection_sequence,
        dtype=bool,
    )

    return landmarks_sequence, detection_sequence

def process_annotation(
    annotation,
    video_root,
    detector,
) -> GestureSequence:
    """Process one annotation row into a GestureSequence."""

    video_path = (
        video_root
        / f"{annotation['video']}.avi"
    )

    landmarks_sequence, detection_sequence = (
        extract_segment_landmarks(
            video_path=video_path,
            start_frame=int(annotation["start_frame"]),
            end_frame=int(annotation["end_frame"]),
            detector=detector,
        )
    )

    normalized_sequence = normalize_landmarks(
        landmarks_sequence
    )

    flattened_sequence = flatten_landmarks(
        normalized_sequence
    )

    return create_gesture_sequence(
        video=annotation["video"],
        label=annotation["label"],
        class_id=int(annotation["class_id"]),
        landmarks=flattened_sequence,
        hand_detected=detection_sequence,
    )