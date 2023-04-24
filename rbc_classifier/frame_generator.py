from background_subtractor import BackgroundSubtractor
import random
from typing import Tuple
import numpy as np
import cv2

TARGET_SHAPE = (132, 800, 3)

class FrameGenerator:
    def __init__(self, batch_size: int = 8, shuffle: bool = True, target_shape: Tuple[int, int, int] = TARGET_SHAPE):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.bg_subtractor = BackgroundSubtractor()

    def __iter__(self) -> 'FrameGenerator':
        if self.shuffle:
            random.shuffle(self.files)
        for i in range(0, len(self.files), self.batch_size):
            batch_files = self.files[i:i + self.batch_size]
            if len(batch_files) < self.batch_size:
                continue
            batch = np.zeros((self.batch_size, self.target_shape[0], self.target_shape[1], self.target_shape[2]))
            for j, file in enumerate(batch_files):
                # Load the AVI file
                with cv2.VideoCapture(file) as cap:
                    video_frames = []
                    while True:
                        # Read a frame from the video
                        ret, frame = cap.read()
                        if ret:
                            # Apply background subtraction
                            frame_processed = self.bg_subtractor.apply(frame)
                            # Update the batch numpy array with the processed frame
                            batch[j, ...] = frame_processed
                        else:
                            break
            # Preprocess the video frames by normalizing
            batch = batch.astype(np.float32) / 255.0
            yield batch


frame_gen = FrameGenerator()