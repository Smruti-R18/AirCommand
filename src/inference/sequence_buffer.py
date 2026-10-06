from collections import deque
import numpy as np

class SequenceBuffer:
    """Store a rolling sequence of frame-level features"""
    def __init__(self,max_length: int):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)

    def append(self,features: np.ndarray) -> None:
        """Add one frame of features to the buffer"""
        self.buffer.append(features)

    def get_sequence(self) -> np.ndarray:
        """Return the current sequence as a Numpy array"""
        if not self.buffer:
            return np.empty((0,0), dtype=np.float32)
        return np.stack(self.buffer,axis=0)

    def clear(self) -> None:
        """Remove all stored features"""
        self.buffer.clear()

    def __len__(self) -> int:
        """Return the number of stored frames"""
        return len(self.buffer)

