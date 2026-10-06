import numpy as np


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize hand landmarks relative to the wrist."""

    wrist = landmarks[0:1, :]
    normalized = landmarks - wrist

    return normalized


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Flatten 21 hand landmarks into 63 features."""

    return landmarks.reshape(-1)


def create_position_features(landmarks: np.ndarray) -> np.ndarray:
    """Create normalized and flattened position features."""

    normalized = normalize_landmarks(landmarks)
    position_features = flatten_landmarks(normalized)

    return position_features

def calculate_motion(
    current_position: np.ndarray,
    previous_position: np.ndarray | None,
) -> np.ndarray:
    """Calculate first-order frame-to-frame motion."""

    if previous_position is None:
        return np.zeros_like(current_position)

    return current_position - previous_position

def create_frame_features(
    landmarks: np.ndarray,
    previous_position: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the 126 features for one frame."""

    position = create_position_features(landmarks)
    motion = calculate_motion(
        current_position=position,
        previous_position=previous_position,
    )

    features = np.concatenate(
        [position, motion]
    )

    return features, position

SHORT_SEQUENCE_LIMIT = 192
LONG_SEQUENCE_LIMIT = 256


def resample_sequence(
    sequence: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample a temporal sequence to a target number of frames."""

    original_length = sequence.shape[0]

    if original_length == target_length:
        return sequence.copy()

    original_positions = np.linspace(
        0,
        original_length - 1,
        num=original_length,
    )

    target_positions = np.linspace(
        0,
        original_length - 1,
        num=target_length,
    )

    resampled = np.empty(
        (target_length, sequence.shape[1]),
        dtype=np.float32,
    )

    for feature_index in range(sequence.shape[1]):
        resampled[:, feature_index] = np.interp(
            target_positions,
            original_positions,
            sequence[:, feature_index],
        )

    return resampled

def prepare_sequence(
    sequence: np.ndarray,
) -> np.ndarray:
    """Prepare a buffered sequence using the training-time hybrid strategy."""

    sequence_length = sequence.shape[0]

    if sequence_length == 0:
        raise ValueError("Cannot prepare an empty sequence.")

    if sequence_length < SHORT_SEQUENCE_LIMIT:
        target_length = sequence_length

    elif sequence_length <= LONG_SEQUENCE_LIMIT:
        target_length = SHORT_SEQUENCE_LIMIT

    else:
        target_length = LONG_SEQUENCE_LIMIT

    return resample_sequence(
        sequence,
        target_length,
    )