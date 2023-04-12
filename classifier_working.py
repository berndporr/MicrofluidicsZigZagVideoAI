import os
import time
import random
import os
import time
import random
import logging
from typing import Tuple

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.applications import EfficientNetB0
from keras.models import Model
from keras.layers import LSTM, TimeDistributed, Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import mixed_precision
from sklearn.utils import shuffle
from tqdm.keras import TqdmCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    logger.info(f"{len(tf.config.list_physical_devices('GPU'))} GPU available")


def main():
    setup_gpu_memory_growth()


if __name__ == '__main__':
    main()


import cv2
import numpy as np


class RBCExtractor:
    def __init__(self, target_shape=(224, 224, 3), min_blob_size=50, max_blob_size=500, min_circularity=0.5):
        self.target_shape = target_shape
        self.min_blob_size = min_blob_size
        self.max_blob_size = max_blob_size
        self.min_circularity = min_circularity
        self.bg_subtractor = BackgroundSubtractor()

    def extract(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)

        # Find blobs in the foreground mask
        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(fg_mask)
        rbc_keypoints = []
        for kp in keypoints:
            # Filter blobs based on size and circularity
            if self.min_blob_size < kp.size < self.max_blob_size and kp.size / kp.angle > self.min_circularity:
                rbc_keypoints.append(kp)

        # Create square regions around the detected RBCs
        regions = []
        for kp in rbc_keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size * 0.8)  # 80% of the original size
            x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
            x2, y2 = min(frame.shape[1] - 1, x + size // 2), min(frame.shape[0] - 1, y + size // 2)
            region = frame[y1:y2, x1:x2, :]
            region = cv2.resize(region, self.target_shape[:2], interpolation=cv2.INTER_AREA)
            regions.append(region)

        # Return the resized regions
        return np.array(regions)


class BackgroundSubtractor:
    def __init__(self):
        self.bg_model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=10)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        # Extract the foreground mask using MOG2 background subtractor
        fg_mask = self.bg_subtractor.apply(gray_frame)

        # Refine the foreground mask
        kernel = np.ones((2, 2), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        fg_mask = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)[1]

        # Update the background model
        if self.bg_model is None:
            self.bg_model = gray_frame.copy().astype(np.float32)
        alpha = 0.3
        self.bg_model = alpha * gray_frame.astype(np.float32) + (1 - alpha) * self.bg_model

        # Create the background mask
        bg_mask = cv2.absdiff(gray_frame.astype(np.float32), self.bg_model.astype(np.float32))
        _, bg_mask = cv2.threshold(bg_mask.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)

        # Combine foreground and background masks
        combined_mask = cv2.bitwise_and(fg_mask, bg_mask)

        # Dilate the combined mask
        kernel_dilation = np.ones((8, 8), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel_dilation, iterations=3)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        # Extract the bounding box of the largest connected component
        contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            x, y, w, h = cv2.boundingRect(contours[max_index])
            bbox = (x, y, w, h)
        else:
            bbox = (0, 0, frame.shape[1], frame.shape[0])

        # Extract the foreground
        fg = cv2.bitwise_and(frame, frame, mask=combined_mask)

        return fg


bg_subtractor = BackgroundSubtractor()


class FrameGenerator:
    def __init__(self, batch_size: int = 16, shuffle: bool = True, target_shape: Tuple[int, int, int] = (224, 224, 3)):
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
                            # Resize the frame and apply background subtraction
                            frame_resized = FrameGenerator.resize_frame(frame, self.target_shape)
                            # Update the batch numpy array with the processed frame
                            batch[j, ...] = frame_resized
                        else:
                            break
            # Preprocess the video frames by normalizing
            batch = batch.astype(np.float32) / 255.0
            yield batch

    @staticmethod
    def resize_frame(frame, target_shape):
        # Crop or pad the frame to make it a square
        h, w = frame.shape[:2]
        if h > w:
            pad_top = 0
            pad_bottom = 0
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
        else:
            pad_left = 0
            pad_right = 0
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top
        preprocessed_frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Resize the framel
        resized_frame = cv2.resize(preprocessed_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

        return resized_frame

frame_gen = FrameGenerator()


frame_gen = FrameGenerator()


def process_videos(video_paths, bg_subtractor, target_shape, max_frames=None):
    processed_videos = []
    labels_list = []

    for video_path in tqdm(video_paths, desc='Processing videos'):
        if any(video_path.startswith(p) for p in healthy_paths):
            label = 1
        elif any(video_path.startswith(p) for p in ill_paths):
            label = 0
        else:
            raise ValueError(f"Video path {video_path} is not in healthy_paths or ill_paths")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue
        video_frames = []
        while True:
            ret, frame = cap.read()
            if ret and frame.size > 0:  # check that frame is not empty:
                preprocessed_frame = bg_subtractor.apply(frame)
                resized_frame = FrameGenerator.resize_frame(preprocessed_frame, target_shape)
                video_frames.append(resized_frame)
                # if not os.path.exists('/home/raj/PycharmProjects/frames'):
                #     os.makedirs('/home/raj/PycharmProjects/frames')
                # cv2.imwrite('/home/raj/PycharmProjects/frames/' + str(label) + str(len(video_frames)).zfill(3) + '_orig.jpg', frame)
                # cv2.imwrite('/home/raj/PycharmProjects/frames/' + str(label) + str(len(video_frames)).zfill(3) + '_bgsub.jpg', preprocessed_frame)
                # cv2.imwrite('/home/raj/PycharmProjects/frames/' + str(label) + str(len(video_frames)).zfill(3) + '_resized.jpg', resized_frame)
            else:
                break
        cap.release()

        if not video_frames:
            continue

        if max_frames is not None:
            num_frames = len(video_frames)
            if num_frames < max_frames:
                padding = np.zeros((max_frames - num_frames, *target_shape[1:]), dtype=np.uint8)
                video_frames.extend([padding] * (max_frames - num_frames))
            elif num_frames > max_frames:
                video_frames = video_frames[:max_frames]

        video_data = np.stack(video_frames, axis=0)

        if label == 1:
            processed_videos.append((video_data, 1, video_data.shape[0]))
        else:
            processed_videos.append((video_data, 0, video_data.shape[0]))

        labels_list.append(label)

    random.shuffle(processed_videos)

    X = np.array([video[0] for video in processed_videos])
    y = np.array([video[1] for video in processed_videos])
    num_frames = np.array([video[2] for video in processed_videos])

    X = X.astype(np.float32) / 255.0

    # Print a random video's label and file path
    idx = np.random.randint(len(processed_videos))
    video_label = processed_videos[idx][1]
    video_path = video_paths[idx]
    print(f"Random video label: {video_label}, file path: {video_path}")

    return X, y, num_frames, np.array(labels_list)

def count_videos(path_list):
    num_videos = 0
    for path in path_list:
        num_videos += len([f for f in os.listdir(path) if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))])
    return num_videos


# Define the healthy and ill paths
healthy_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused", ]

ill_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused", ]

num_healthy_videos = count_videos(healthy_paths)
num_ill_videos = count_videos(ill_paths)
print(f"{num_healthy_videos} healthy videos in the server")
print(f"{num_ill_videos} ill videos in the server")

# Set the train and val split percentages
train_split = 0.7
val_split = 0.3


# Define a function to split videos into train and validation sets
def split_videos(path_list, train_split):
    train_paths = []
    val_paths = []
    for path in path_list:
        videos = [f for f in os.listdir(path) if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))]
        random.shuffle(videos)
        num_train = int(train_split * len(videos))
        train_paths.extend([os.path.join(path, v) for v in videos[:num_train]])
        val_paths.extend([os.path.join(path, v) for v in videos[num_train:]])
    return train_paths, val_paths


# Split healthy and ill videos into train and validation sets
healthy_train, healthy_val = split_videos(healthy_paths, train_split)
ill_train, ill_val = split_videos(ill_paths, train_split)

# Limit the dataset to 1500 healthy/ill train and 250 healthy/ill val
healthy_train = random.sample(healthy_train, 100)
ill_train = random.sample(ill_train, 100)

healthy_val = random.sample(healthy_val, 5)
ill_val = random.sample(ill_val, 5)

print(f"{len(healthy_train)} healthy training videos sent to the model")
print(f"{len(ill_train)} ill training videos sent to the model")
time.sleep(1)
print(f"{len(healthy_val)} healthy validation videos sent to the model")
print(f"{len(ill_val)} ill validation videos sent to the model")
time.sleep(1)

# Define the target shape of the resized video frames
target_shape = (224, 224, 3)


def preprocess_data(healthy_data, ill_data, target_shape, num_samples):
    # Resize and preprocess the healthy and ill data, and get labels
    healthy_data_resized, _, num_healthy_frames, healthy_labels = process_videos(healthy_data, bg_subtractor, target_shape)
    ill_data_resized, _, num_ill_frames, ill_labels = process_videos(ill_data, bg_subtractor, target_shape)

    # Stack the healthy and ill data vertically
    data_resized = np.vstack([healthy_data_resized, ill_data_resized])
    data_resized = data_resized.astype(np.float32)
    data_resized = data_resized[:num_samples]

    # Use the labels from process_videos()
    labels = np.concatenate([healthy_labels, ill_labels])
    labels = labels[:num_samples]

    return data_resized, labels

# Preprocess the training data
train_data_resized, train_labels = preprocess_data(healthy_train, ill_train, target_shape=(224, 224, 3), num_samples=4)

# Preprocess the validation data
val_data_resized, val_labels = preprocess_data(healthy_val, ill_val, target_shape=(224, 224, 3), num_samples=4)

print("train_data_resized shape:", train_data_resized.shape, "dtype:", train_data_resized.dtype)
print("train_labels shape:", train_labels.shape, "dtype:", train_labels.dtype)
print("val_data_resized shape:", val_data_resized.shape, "dtype:", val_data_resized.dtype)
print("val_labels shape:", val_labels.shape, "dtype:", val_labels.dtype)





# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Add a GlobalAveragePooling2D layer to convert 4D tensor to 3D tensor
global_avg_pool = GlobalAveragePooling2D()(base_model.output)

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

inputs = Input(shape=(250, 224, 224, 3))
time_distributed_base_model = TimeDistributed(base_model)(inputs)

x = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))(time_distributed_base_model)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
x = TimeDistributed(Flatten())(x)

# Pass the 3D tensor to the LSTM layer
x = LSTM(units=64, activation='relu')(x)
predictions = Dense(units=1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=inputs, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Define batch size and number of epochs
epochs = 5
batch_size = 32

# Train the model with tqdm progress bars
history = model.fit(
    train_data_resized,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(val_data_resized, val_labels),
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_lr=0.0001),
        TqdmCallback(verbose=1)
    ],
    verbose=0
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_data_resized, val_labels, verbose=0)
print('Validation loss:', val_loss)
print('Validation accuracy:', val_acc)

# Print model summary and trainable parameters
print(model.summary(include_all=False))
print("Trainable parameters:", model.count_params())