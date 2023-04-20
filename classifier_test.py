import os
import random
import cv2
import numpy as np
import tensorflow as tf
import logging
from typing import Tuple

from keras.applications import EfficientNetB0
from tqdm import tqdm
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.optimizers import Adam

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def setup_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


setup_gpu_memory_growth()


class BackgroundSubtractor:
    def __init__(self):
        self.bg_model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=10)
        # createBackgroundSubtractorKNN(history=50, dist2Threshold=800)
        self.previous_fg_mask = None
        # decrease history, increase threshold

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to remove noise
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)
        # larger kernal = aggressive smoothing

        # Apply background subtraction using temporal difference
        fg_mask = self.bg_subtractor.apply(gray_frame)

        # Apply thresholding to remove low values and keep high values (moving objects)
        _, fg_mask = cv2.threshold(fg_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply erosion and dilation to remove small blobs and fill holes
        kernel = np.ones((2, 2), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

        # Apply background subtraction using spatial difference
        if self.bg_model is None:
            self.bg_model = gray_frame.copy().astype(np.float32)
        alpha = 0.3  # Increased alpha value
        self.bg_model = alpha * gray_frame.astype(np.float32) + (1 - alpha) * self.bg_model
        bg_mask = cv2.absdiff(gray_frame.astype(np.float32), self.bg_model.astype(np.float32))
        _, bg_mask = cv2.threshold(bg_mask.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)  # Increased threshold value

        # Combine temporal and spatial difference masks
        combined_mask = cv2.bitwise_and(fg_mask, bg_mask)

        # Apply dilation to fill the RBC region
        kernel_dilation = np.ones((5, 5), np.uint8)  # Increase the kernel size for dilation
        combined_mask = cv2.dilate(combined_mask, kernel_dilation, iterations=3)  # Increase the number of iterations

        # Apply erosion and dilation to remove small blobs and fill holes
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        # Apply mask to original frame to extract moving objects
        fg = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Blend the current frame with the previous frame(s)
        if self.previous_fg_mask is not None:
            fg = cv2.addWeighted(fg, 0.5, self.previous_fg_mask, 0.5, 0)

        # Update the previous_fg_mask
        self.previous_fg_mask = fg.copy()

        # Remove static background
        bg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(combined_mask))

        # Return the foreground image
        return fg


bg_subtractor = BackgroundSubtractor()


class FrameGenerator:
    def __init__(self, batch_size: int = 16, shuffle: bool = True, target_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_shape = target_shape
        self.bg_subtractor = BackgroundSubtractor()

    @staticmethod
    def resize_frame(frame: np.ndarray, target_shape: Tuple[int, int, int],
                     bg_subtractor: BackgroundSubtractor) -> np.ndarray:
        # Extract the first element from the tuple if the frame is a tuple
        if isinstance(frame, tuple):
            frame = frame[0]
        # Apply background subtraction to the frame
        fg = bg_subtractor.apply(frame)

        # Get dimensions of the frame
        h, w = fg.shape[:2]

        # Calculate the scaling factor while preserving aspect ratio
        scale = min(target_shape[0] / h, target_shape[1] / w)
        h_scaled, w_scaled = round(h * scale), round(w * scale)

        # Resize the frame while preserving aspect ratio
        fg_resized = cv2.resize(fg, (w_scaled, h_scaled))

        # Calculate padding dimensions to create a square image
        padding_h1 = (target_shape[0] - h_scaled) // 2
        padding_h2 = target_shape[0] - h_scaled - padding_h1
        padding_w1 = (target_shape[1] - w_scaled) // 2
        padding_w2 = target_shape[1] - w_scaled - padding_w1

        # Apply padding to the frame
        padded_fg = cv2.copyMakeBorder(fg_resized, padding_h1, padding_h2, padding_w1, padding_w2, cv2.BORDER_CONSTANT,
                                       value=0)

        return padded_fg

    def __iter__(self) -> 'FrameGenerator':
        if self.shuffle:
            random.shuffle(self.files)
        for i in range(0, len(self.files), self.batch_size):
            batch_files = self.files[i:i + self.batch_size]
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
                            frame_resized = self.resize_frame(frame, self.target_shape, self.bg_subtractor)
                            # Update the batch numpy array with the processed frame
                            batch[j, ...] = frame_resized
                        else:
                            break
            # Preprocess the video frames by normalizing
            batch = batch.astype(np.float32) / 255.0
            yield batch


frame_gen = FrameGenerator()


def process_videos(paths, bg_subtractor, num_videos=int, skip_frames=0):
    processed_videos = []
    for path in paths:
        files = os.listdir(path)
        avi_files = [file for file in files if file.endswith('.avi')]
        random.shuffle(avi_files)
        for file in tqdm(avi_files[:num_videos], desc="Processing videos"):
            # Load the AVI file
            cap = cv2.VideoCapture(os.path.join(path, file))
            video_frames = []
            frame_count = 0
            while True:
                # Read a frame from the video
                ret, frame = cap.read()
                if ret:
                    if frame_count % (skip_frames + 1) == 0:
                        fg = bg_subtractor.apply(frame)

                        # # Save the original frame to disk
                        # cv2.imwrite(
                        #     f"/home/raj/PycharmProjects/droplets_video/save/{file[:-4]}_{frame_count}_orig.jpg", frame)

                        # Save the frame after applying background subtraction to disk
                        # if fg.shape[0] > 0 and fg.shape[1] > 0:
                        #     cv2.imwrite(
                        #         f"/home/raj/PycharmProjects/droplets_video/save/{file[:-4]}_{frame_count}_bgsub.png",
                        #         fg)
                        #     print("saved")

                        # Resize the frame
                        frame_resized = FrameGenerator.resize_frame(fg, target_shape=(224, 224, 3),
                                                                    bg_subtractor=bg_subtractor)

                        # Save the frame after resizing to disk
                        # cv2.imwrite(
                        #     f"/home/raj/PycharmProjects/droplets_video/save/{file[:-4]}_{frame_count}_resized.jpg",
                        #     frame_resized)

                        # Append the processed frame to the video frames list
                        video_frames.append(frame_resized)

                        # # Save the processed frame to disk
                        # cv2.imwrite(
                        #     f"/home/raj/PycharmProjects/droplets_video/saved/{file[:-4]}_{frame_count}_processed.jpg",
                        #     frame_resized)

                        frame_count += 1
                    else:
                        frame_count += 1
                else:
                    break
            # Release the VideoCapture object
            cap.release()
            # Stack the video frames to form a 3D tensor of shape (num_frames, height, width, channels)
            video_tensor = np.stack(video_frames)
            processed_videos.append(video_tensor)

    return np.concatenate(processed_videos, axis=0)


healthy_train_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                       "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused"]
healthy_train_data_resized = process_videos(healthy_train_paths, bg_subtractor, num_videos=10)

ill_train_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
                   "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused"]
ill_train_data_resized = process_videos(ill_train_paths, bg_subtractor, num_videos=10)

healthy_val_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]
healthy_val_data_resized = process_videos(healthy_val_paths, bg_subtractor, num_videos=5)

ill_val_paths = ["/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]
ill_val_data_resized = process_videos(ill_val_paths, bg_subtractor, num_videos=5)


def train_model(model, healthy_train_data, ill_train_data, healthy_val_data, ill_val_data, epochs, batch_size):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch in tqdm(range(0, healthy_train_data.shape[0], batch_size)):
            healthy_x_batch = healthy_train_data[batch:batch + batch_size]
            ill_x_batch = ill_train_data[batch:batch + batch_size]

            x_batch = np.concatenate([healthy_x_batch, ill_x_batch], axis=0)
            y_batch = np.concatenate([np.zeros(healthy_x_batch.shape[0]), np.ones(ill_x_batch.shape[0])], axis=0)

            model.train_on_batch(x_batch, y_batch)

        train_loss, train_acc = model.evaluate(x=np.concatenate([healthy_train_data_resized, ill_train_data_resized], axis=0),
                                               y=np.concatenate([np.zeros(healthy_train_data_resized.shape[0]),
                                                                 np.ones(ill_train_data_resized.shape[0])], axis=0))
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Train loss: {train_loss:.1f}")
        print(f"Train accuracy: {train_acc:.1f}")

        val_data = np.concatenate([healthy_val_data_resized, ill_val_data_resized], axis=0)
        val_labels = np.concatenate([np.zeros(healthy_val_data_resized.shape[0]), np.ones(ill_val_data_resized.shape[0])], axis=0)

        val_loss, val_acc = model.evaluate(x=val_data, y=val_labels)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Validation loss: {val_loss:.1f}")
        print(f"Validation accuracy: {val_acc:.1f}")

    return train_losses, train_accuracies, val_losses, val_accuracies, model


# Load pre-trained EfficientNetB0 model
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(units=64, activation='relu')(x)
predictions = Dense(units=1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
batch_size = 32

train_losses, train_accuracies, val_losses, val_accuracies, trained_model = train_model(model,
                                                                                        healthy_train_data_resized,
                                                                                        ill_train_data_resized,
                                                                                        healthy_val_data_resized,
                                                                                        ill_val_data_resized,
                                                                                        epochs,
                                                                                        batch_size)

# print("")
# print(f"Train losses: {train_losses}")
# print(f"Train accuracies: {train_accuracies}")
# print(f"Validation losses: {val_losses}")
# print(f"Validation accuracies: {val_accuracies}")

# healthy_val_labels = np.zeros(len(healthy_val_data_resized))
# ill_val_labels = np.ones(len(ill_val_data_resized))
#
# val_data_resized = np.concatenate((healthy_val_data_resized, ill_val_data_resized), axis=0)
# val_labels = np.concatenate((healthy_val_labels, ill_val_labels), axis=0)

# # Evaluate the model on the test set
# loss, accuracy = model.evaluate(val_data_resized)
# print('Val loss:', loss)
# print('Val accuracy:', accuracy)

# # Print model summary and trainable parameters
# print(trained_model.summary())
# print("Trainable parameters:", sum([param.trainable for param in trained_model.trainable_weights]))
