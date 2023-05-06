import logging
from tqdm import tqdm
import cv2
import random
import numpy as np
import os
import tensorflow as tf


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


def process_dataset(native_videos, modified_videos, native_labels, modified_labels):
    # Process the native videos.
    processed_native_videos = process_videos(native_videos)
    processed_modified_videos = process_videos(modified_videos)

    # Concatenate the native and modified data.
    processed_videos = np.concatenate([processed_native_videos, processed_modified_videos], axis=0)
    processed_videos = processed_videos.astype(np.float32) / 255.0
    processed_videos = tf.data.Dataset.from_tensor_slices(processed_videos)

    # # Print processed_videos shape
    # print(processed_videos.element_spec)

    # Use the provided labels.
    labels = np.concatenate([native_labels, modified_labels], axis=0)
    labels = labels.astype(np.int16)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    # # Print processed_videos shape
    # print(labels.element_spec)

    # Return the processed data and labels.
    return processed_videos, labels


def process_videos(videos):
    # Initialize variables.
    processed_videos = []
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)

    # Iterate over the video paths.
    for video_path in tqdm(videos, desc='Processing videos', position=0, leave=True):
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

            # Skip the frame if it could not be read.
            if not ret or frame.size == 0:
                break

            # Apply background subtraction to the frame.
            fgMask = bgSub.apply(frame)

            # Convert the foreground mask to RGB.
            mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)

            # Apply the mask to the frame.
            processed_frame = cv2.bitwise_and(frame, mask)

            # Skip the first and last 10 frames.
            if 50 < frame_count <= 150:
                # Skip every other frame.
                if frame_count % 2 == 0:
                    video_frames.append(processed_frame)

                    # # Save the frame.
                    # cv2.imwrite(f'/home/raj/PycharmProjects/frames/{frame_count:03}_processed.jpg', processed_frame)

        # Close the video file.
        cap.release()

        # If there are no frames in the video, skip it.
        if len(video_frames) == 0:
            continue

        # Stack the frames into a NumPy array.
        video_frames = np.array(video_frames)[..., [2, 1, 0]]
        video_frames = np.maximum(video_frames, 0)
        processed_videos.append(np.stack(video_frames, axis=0))

    video_data = processed_videos

    # Return the processed videos.
    return video_data
