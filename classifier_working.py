import logging
import os
import random
import time
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import sklearn.utils as sk_utils
import tensorflow as tf
from tqdm import tqdm
from keras.utils import to_categorical
from keras import layers
from keras import mixed_precision
from keras.applications import EfficientNetB0, MobileNet, MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import (Bidirectional, Concatenate, Conv2D, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, Input, LSTM,
                          MaxPooling2D, TimeDistributed)
from keras.losses import BinaryCrossentropy
from keras.metrics import Accuracy
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops.custom_gradient import recompute_grad

# Disable all logging messages
logging.disable(logging.CRITICAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def setup_gpu_memory_growth() -> None:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    print(f"{len(physical_devices)} GPU(s) available")


def main():
    setup_gpu_memory_growth()
    if tf.test.is_gpu_available():
        print("TensorFlow is running on GPU")
        print("")
    else:
        print("TensorFlow is running on CPU")
        print("")


if __name__ == '__main__':
    main()


class BackgroundSubtractor:
    def __init__(self):
        self.bg_model = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=10)
        self.bbox = None

        #  if the background is relatively static,
        #  a higher value of history may be appropriate to provide better background modeling.
        #  if the video stream contains a lot of noise or the background changes frequently,
        #  a lower value for varThreshold might be more appropriate to avoid false detections.

        # Initialize CLAHE object
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = self.clahe.apply(gray_frame)  # Apply CLAHE to improve contrast
        gray_frame = cv2.GaussianBlur(gray_frame, (3, 3), 0)

        # Apply a slight sharpening effect to the image
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9, -1],
                                      [-1, -1, -1]])
        gray_frame = cv2.filter2D(gray_frame, -1, kernel_sharpening)

        # Extract the foreground mask using KNN background subtractor
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
        kernel_dilation = np.ones((7, 7), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel_dilation, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            x, y, w, h = cv2.boundingRect(contours[max_index])
            self.bbox = (x, y, w, h)
        else:
            self.bbox = (0, 0, frame.shape[1], frame.shape[0])

        # Extract the foreground
        fg = cv2.bitwise_and(frame, frame, mask=combined_mask)

        return fg


bg_subtractor = BackgroundSubtractor()

# Define the target shape of the resized video frames
target_shape = (224, 224, 3)
num_frames = 10


class FrameGenerator:
    def __init__(self, batch_size: int = 8, shuffle: bool = True, target_shape: Tuple[int, int, int] = target_shape):
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
    def resize_and_center_frame(frame, target_shape, bbox):
        h, w = frame.shape[:2]
        max_dim = max(h, w)
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
        x, y, w, h = bbox

        # Crop the region of interest
        roi = frame[y:y + h, x:x + w]

        # Calculate padding for the cropped frame to make it square
        if h > w:
            pad_left = (h - w) // 2
            pad_right = h - w - pad_left
        else:
            pad_top = (w - h) // 2
            pad_bottom = w - h - pad_top

        # Pad the cropped frame
        padded_roi = cv2.copyMakeBorder(roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # Resize the padded_roi to fill 100% of the target_shape
        scale = min(target_shape[0] * 1.0 / h, target_shape[1] * 1.0 / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_roi = cv2.resize(padded_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Pad the resized_roi to the final target_shape
        pad_top = (target_shape[0] - new_h) // 2
        pad_bottom = target_shape[0] - new_h - pad_top
        pad_left = (target_shape[1] - new_w) // 2
        pad_right = target_shape[1] - new_w - pad_left
        final_frame = cv2.copyMakeBorder(resized_roi, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        return final_frame

    @staticmethod
    def resize_frame(frame, target_shape, bbox=None):
        if bbox is not None:
            # Apply the original resizing
            preprocessed_frame = FrameGenerator.resize_and_center_frame(frame, target_shape, bbox)
        return preprocessed_frame


frame_gen = FrameGenerator()


def process_videos(video_paths_labels, bg_subtractor, target_shape, max_frames=None):
    processed_videos = []
    labels_list = []
    discarded_videos_count = 0

    for video_path, label in tqdm(video_paths_labels, desc='Processing videos', position=0, leave=True):
        # tqdm.write(f"Currently processing video {video_path}", end='\r')

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue

        video_frames = []
        frame_count = 0
        empty_frames_count = 0
        non_empty_frames_count = 0

        # Iterate over the frames and select frames in the range 40-160 and skip every other frame
        while True:
            ret, frame = cap.read()
            if not ret or frame.size == 0:
                break

            frame_count += 1

            # Skip frames that are not in the range of interest
            if frame_count < 40:
                continue
            elif 40 <= frame_count < 160:
                # Skipping frames
                if (frame_count - 40) % 6 == 0:
                    preprocessed_frame = bg_subtractor.apply(frame)

                    # Check if the frame is empty after background subtraction
                    if np.sum(preprocessed_frame) == 0:
                        empty_frames_count += 1
                    else:
                        non_empty_frames_count += 1

                    # If there are more than 50 empty frames, skip the video
                    if empty_frames_count > 50:
                        discarded_videos_count += 1
                        break

                    bbox = bg_subtractor.bbox
                    resized_frame = FrameGenerator.resize_frame(preprocessed_frame, target_shape, bbox)
                    video_frames.append(resized_frame)

                    if not os.path.exists('/home/raj/PycharmProjects/frames'):
                        os.makedirs('/home/raj/PycharmProjects/frames')
                    cv2.imwrite('/home/raj/PycharmProjects/frames/'
                                + str(label) + str(len(video_frames)).zfill(3) + '_orig.jpg', frame)
                    # cv2.imwrite('/home/raj/PycharmProjects/frames/'
                    # + str(label) + str(len(video_frames)).zfill(3) + '_bgsub.jpg', preprocessed_frame)
                    cv2.imwrite('/home/raj/PycharmProjects/frames/'
                                + str(label) + str(len(video_frames)).zfill(3) + '_resized.jpg', resized_frame)
                    # Save the resized frame as an array
                    # np.save(f'/home/raj/PycharmProjects/array/'
                    # f'{os.path.basename(video_path)}_{label}_{len(video_frames):03}.npy', resized_frame)

            elif frame_count >= 160:
                break
        cap.release()

        # Discard the first and last 5 frames
        video_frames = video_frames[5:-5]

        # Skip videos that do not have RBC flowing through the channel
        if len(video_frames) == 0:
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

    # # Print the count of discarded videos
    # print(f"{discarded_videos_count} empty videos")

    # random.shuffle(processed_videos)

    X = np.array([video[0] for video in processed_videos])
    y = np.array([video[1] for video in processed_videos])
    num_frames = np.array([video[2] for video in processed_videos])

    X = X.astype(np.float32) / 255.0

    # Print a random video's label and file path
    idx = np.random.randint(len(processed_videos))
    video_label = processed_videos[idx][1]
    video_path = video_paths_labels[idx]
    print(f"{video_path}")

    return X, y, num_frames, np.array(labels_list)


# ----------------------------------- #

# Limit the dataset
# num = int(input("Enter the number of videos: "))
# num = 50
train_videos = 450
    # int(num * 0.9)
val_videos = 50
    # int(num * 0.1)

print(f"{int(train_videos)} training videos")
print(f"{int(val_videos)} validation videos")
print("")
# time.sleep(1)

# Define the healthy and ill paths
healthy_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused",
                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_overfocused2ticks",
                 "/data/RBC_Phantom_60xOlympus/Donor_1/Native5_underfocused2ticks",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_3_focused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_2_underfocused",
                 "/data/RBC_Phantom_60xOlympus/Donor_2/RBC_9March2023_Donor2_4_overfocused"]

ill_paths = ["/data/RBC_Phantom_60xOlympus/Donor_1/FA_0.37wtPercent",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_focused",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Overfocused",
             "/data/RBC_Phantom_60xOlympus/Donor_2/RBC10March2023_Donor2_2ndDay_1mMDiamide_Split_Underfocused"]


def get_videos(path_list, label, num_videos):
    video_paths = []
    video_labels = []
    all_videos = []

    for path in path_list:
        videos = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))]
        all_videos.extend(videos)

    random.shuffle(all_videos)
    selected_videos = all_videos[:num_videos]

    for video in selected_videos:
        video_paths.append(video)
        video_labels.append(label)

    return video_paths, video_labels


# Select videos for healthy and ill training and validation sets
healthy_train_paths, healthy_train_labels = get_videos(healthy_paths, label=1, num_videos=train_videos // 2)
ill_train_paths, ill_train_labels = get_videos(ill_paths, label=0, num_videos=train_videos // 2)

healthy_val_paths, healthy_val_labels = get_videos(healthy_paths, label=1, num_videos=val_videos // 2)
ill_val_paths, ill_val_labels = get_videos(ill_paths, label=0, num_videos=val_videos // 2)

print(f"{len(healthy_train_paths)} healthy training videos sent to the model")
print(f"{len(ill_train_paths)} ill training videos sent to the model")
# time.sleep(1)
print("")

print(f"{len(healthy_val_paths)} healthy validation videos sent to the model")
print(f"{len(ill_val_paths)} ill validation videos sent to the model")
# time.sleep(1)
print("")


def preprocess_data(healthy_data, ill_data, healthy_labels, ill_labels, target_shape, num_samples):
    healthy_video_paths_labels = list(zip(healthy_data, healthy_labels))
    ill_video_paths_labels = list(zip(ill_data, ill_labels))

    healthy_data_resized, _, num_healthy_frames, _ = process_videos(healthy_video_paths_labels, bg_subtractor, target_shape)
    ill_data_resized, _, num_ill_frames, _ = process_videos(ill_video_paths_labels, bg_subtractor, target_shape)

    # Concatenate the healthy and ill data
    data_resized = np.concatenate([healthy_data_resized, ill_data_resized], axis=0)

    # Get the number of input videos
    num_videos = data_resized.shape[0]

    # Reshape the data to the format (num_videos, num_frames, frame_height, frame_width, num_channels)
    data_resized = data_resized.reshape(num_videos, num_frames, *target_shape)
    data_resized = data_resized[:num_samples].astype(np.float32)

    # Use the provided labels
    labels = np.concatenate([healthy_labels, ill_labels])
    labels = labels[:num_samples]

    return data_resized, labels


# Preprocess the training data
train_data_resized, train_labels = preprocess_data(healthy_train_paths, ill_train_paths,
                                                   healthy_train_labels, ill_train_labels,
                                                   target_shape=target_shape, num_samples=train_videos)

# Preprocess the validation data
val_data_resized, val_labels = preprocess_data(healthy_val_paths, ill_val_paths,
                                               healthy_val_labels, ill_val_labels,
                                               target_shape=target_shape, num_samples=val_videos)

train_paths = healthy_train_paths + ill_train_paths
val_paths = healthy_val_paths + ill_val_paths

print("")
time.sleep(1)
print("train_data_resized shape:", train_data_resized.shape, "dtype:", train_data_resized.dtype)
print("train_labels shape:", train_labels.shape, "dtype:", train_labels.dtype)
print("")
# time.sleep(1)

# Print 5 random training videos and labels
print("Random training videos and labels:")
for _ in range(5):
    idx = np.random.choice(train_videos)
    label = train_labels[idx]
    video_path = train_paths[idx]
    print("Label:", label, "Video path:", video_path)
print("")
time.sleep(1)

print("val_data_resized shape:", val_data_resized.shape, "dtype:", val_data_resized.dtype)
print("val_labels shape:", val_labels.shape, "dtype:", val_labels.dtype)
print("")
# time.sleep(1)


# Print 5 random validation videos and labels
print("Random validation videos and labels:")
for _ in range(5):
    idx = np.random.choice(val_videos)
    label = val_labels[idx]
    video_path = val_paths[idx]
    print("Label:", label, "Video path:", video_path)
print("")
time.sleep(1)

# ----------------------------------- #

# Define the input shape of the video frames
num_frames = 10
frame_height = 224
frame_width = 224
num_channels = 3

inputs = tf.keras.layers.Input(shape=(num_frames, frame_height, frame_width, num_channels))

# Load the EfficientNetB0 model without the top layer
net = tf.keras.applications.EfficientNetB0(include_top=False)
net.trainable = False

# Add custom layers for classification
x = tf.keras.layers.TimeDistributed(net)(inputs)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(10, activation='relu')(x)
x = tf.keras.layers.GlobalAveragePooling3D()(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

# Finalize the model
model = tf.keras.models.Model(inputs=inputs, outputs=predictions)

# Set the optimizer with a smaller learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)

# Compile the model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 1
epochs = 1

train_labels = to_categorical(train_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)

train_ds = tf.data.Dataset.from_tensor_slices((train_data_resized, train_labels)).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((val_data_resized, val_labels)).batch(batch_size)

# Train the model
history = model.fit(train_ds, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=val_ds)
print("")
accuracy = history.history['accuracy'][0] * 100
print('Accuracy: {:.2f}%'.format(accuracy))

# # policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
# # tf.keras.mixed_precision.experimental.set_policy(policy)
#
# frame_height = 224
# frame_width = 224
# num_channels = 3
# num_classes = 2
#
# # Load the MobileNetV2 model without the top layer
# base_model = MobileNetV2(input_shape=(frame_height, frame_width, num_channels), include_top=False, weights='imagenet')
#
# # Enable fine-tuning of the pre-trained MobileNetV2 model
# base_model.trainable = True
#
# # Add custom layers
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# mobilenet_model = Model(inputs=base_model.input, outputs=x)
#
# # Use TimeDistributed with the MobileNetV2 model
# input_shape = (num_frames, frame_height, frame_width, num_channels)
# inputs = Input(shape=input_shape)
# x = TimeDistributed(mobilenet_model)(inputs)
#
# # Add LSTM layer for sequence processing
# x = LSTM(1024, return_sequences=False)(x)
# x = Dropout(0.5)(x)
#
# # Add dense layers and final softmax layer for classification
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(num_classes, activation='softmax')(x)
#
# # Finalize the model
# model = Model(inputs=inputs, outputs=predictions)
#
# # ----------------------------------- #
#
# # Define the optimizer, loss function, and gradient accumulation steps
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# loss_fn = tf.keras.losses.binary_crossentropy
# accumulation_steps = 2
#
# epochs = 50
# batch_size = 8
#
# # Define the early stopping callback
# # early_stopping_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
#
# # Define the model checkpoint callback
# model_checkpoint_callback = ModelCheckpoint(
#     filepath="/home/raj/PycharmProjects/weights/best_model_weights.h5",
#     monitor="val_accuracy",
#     mode="max",
#     save_best_only=True,
#     save_weights_only=True,
#     verbose=1
# )
#
#
# # Create a function for the train step with gradient accumulation
# @tf.function
# def train_step(images, labels):
#     with tf.GradientTape() as tape:
#         predictions = model(images, training=True)
#         loss_value = loss_fn(labels, predictions)
#     gradients = tape.gradient(loss_value, model.trainable_variables)
#     gradients = [g / tf.constant(float(accumulation_steps)) for g in gradients]
#     return loss_value, gradients
#
#
# # def data_augmentation(video, label):
# #     def augment_frame(frame):
# #         frame = tf.image.random_flip_left_right(frame)
# #         frame = tf.image.random_brightness(frame, max_delta=0.05)
# #         frame = tf.image.random_contrast(frame, lower=0.9, upper=1.1)
# #         return frame
# #
# #     video = tf.map_fn(augment_frame, video)
# #     return video, label
#
# # train_dataset_augmented = train_dataset.map(data_augmentation)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((train_data_resized, train_labels)). \
#     batch(batch_size, drop_remainder=True).shuffle(buffer_size=1024).prefetch(buffer_size=tf.data.AUTOTUNE)
# val_dataset = tf.data.Dataset.from_tensor_slices((val_data_resized, val_labels)). \
#     batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
#
# # # Replace train_dataset with train_dataset_augmented in the training loop
# # for step, (images, labels) in enumerate(train_dataset):
# #     train_dataset = tf.data.Dataset.from_tensor_slices((train_data_resized, train_labels)).batch(batch_size)
#
# # Custom training loop
# train_accuracies = []
# val_accuracies = []
#
# best_val_accuracy = 0
# patience_counter = 0
# accuracy_threshold = 0.7
#
# for epoch in tqdm(range(epochs)):
#     epoch_loss_avg = tf.keras.metrics.Mean()
#     epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#
#     for step, (images, labels) in enumerate(train_dataset):
#         loss_value, gradients = train_step(images, labels)
#
#         if (step + 1) % accumulation_steps == 0:
#             optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#             gradients = [tf.zeros_like(g) for g in gradients]
#
#         epoch_loss_avg.update_state(loss_value)
#         epoch_accuracy.update_state(labels, model(images, training=False))
#
#     val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#     for val_images, val_labels in val_dataset:
#         val_predictions = model(val_images, training=False)
#         val_accuracy.update_state(val_labels, val_predictions)
#
#     current_val_accuracy = val_accuracy.result().numpy()
#
#     print("Epoch {:02d}: Loss: {:.2f}, Train Accuracy: {:.1%}, Validation Accuracy: {:.1%}".format(
#         epoch + 1, epoch_loss_avg.result(), epoch_accuracy.result(), current_val_accuracy))
#
#     # Check for early stopping
#     if current_val_accuracy > best_val_accuracy:
#         best_val_accuracy = current_val_accuracy
#         patience_counter = 0
#
#         # Save model weights if the validation accuracy is greater than the threshold
#         if best_val_accuracy > accuracy_threshold:
#             model.save_weights(model_checkpoint_callback.filepath)
#             print(f"Model weights saved to {model_checkpoint_callback.filepath}")
#
#     else:
#         patience_counter += 1
#
#     if patience_counter >= 10:
#         print("Early stopping triggered: validation accuracy did not improve for {} epochs".format(patience_counter))
#         break
#
# # Load the best weights (achieved during training) into the model
# model.load_weights(model_checkpoint_callback.filepath)
#
# # print("Training accuracies:", train_accuracies)
# # print("Validation accuracies:", val_accuracies)
