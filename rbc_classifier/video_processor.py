import logging
from tqdm import tqdm
import cv2
import random
from background_subtractor import BackgroundSubtractor
import numpy as np
import os
import tensorflow as tf
import time


def get_videos(path_list, label, num_videos):
    # Initialize variables.
    video_paths = []
    video_labels = []
    all_videos = []

    # Iterate over the paths.
    for path in path_list:
        # Get a list of video paths in the directory.
        videos = [os.path.join(path, f)
                  for f in os.listdir(path)
                  if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))]

        # Add the video paths to the list of all videos.
        all_videos.extend(videos)

    # Shuffle the list of all videos.
    random.shuffle(all_videos)

    # Get the first `num_videos` videos from the list of all videos.
    selected_videos = all_videos[:num_videos]

    # Iterate over the selected videos.
    for video in selected_videos:
        # Add the video path and label to the lists of video paths and labels.
        video_paths.append(video)
        video_labels.append(label)

    # Return the lists of video paths and labels.
    return video_paths, video_labels


def process_dataset (healthy_data, ill_data, healthy_labels, ill_labels):
    # Process the videos.
    healthy_data_processed = process_videos(healthy_data)
    ill_data_processed = process_videos(ill_data)

    # Concatenate the healthy and ill data.
    processed_videos = np.concatenate([healthy_data_processed, ill_data_processed], axis=0)
    processed_videos = processed_videos.astype(np.float32) / 255.0
    processed_videos = tf.data.Dataset.from_tensor_slices(processed_videos)

    # Use the provided labels.
    labels = np.concatenate([healthy_labels, ill_labels], axis=0)
    labels = labels.astype(np.int16)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    # Return the processed data and labels.
    return processed_videos, labels


def process_videos(video_paths_labels):
    # Initialize variables.
    processed_videos = []
    bg_subtractor = BackgroundSubtractor()

    # Iterate over the video paths and labels.
    for video_path in tqdm(video_paths_labels, desc='Processing videos', position=0, leave=True):
        # Open the video file.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue

        # Get the number of frames in the video.
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize the video frames list.
        video_frames = []

        # Iterate over the frames in the video.
        for frame_count in range(num_frames):
            ret, frame = cap.read()
            if not ret or frame.size == 0:
                break

            # Apply background subtraction to the frame.
            if 40 < frame_count <= 160:
                preprocessed_frame = bg_subtractor.apply(frame)

                # Skip the first and last 10 frames.
                if 50 < frame_count <= 150:
                    video_frames.append(preprocessed_frame)
                    # cv2.imwrite(f'/home/raj/PycharmProjects/frames/
                    # {label}_{frame_count:03}_preprocessed.jpg', preprocessed_frame)

        # Close the video file.
        cap.release()

        # If there are no frames in the video, skip it.
        if len(video_frames) == 0:
            continue

        # Stack the frames into a NumPy array.
        processed_videos.append(np.stack(video_frames, axis=0))

    video_data = processed_videos

    # Return the processed videos, labels, number of frames, and labels.
    return video_data
