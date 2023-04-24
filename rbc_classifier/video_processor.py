import logging
from tqdm import tqdm
import cv2
import random
from background_subtractor import BackgroundSubtractor
import numpy as np
import os

NUM_FRAMES = 100
bg_subtractor = BackgroundSubtractor()

def process_videos(video_paths_labels, bg_subtractor, target_shape, max_frames=None):
    processed_videos = []
    labels_list = []

    for video_path, label in tqdm(video_paths_labels, desc='Processing videos', position=0, leave=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue

        video_frames = []

        for frame_count in range(160):
            ret, frame = cap.read()
            if not ret or frame.size == 0:
                break

            if 39 < frame_count < 160:
                preprocessed_frame = bg_subtractor.apply(frame)

                if frame_count > 49 and frame_count < 150:  # Skip first and last 10 frames
                    video_frames.append(preprocessed_frame)
                    # cv2.imwrite(f'/home/raj/PycharmProjects/frames/{label}_{frame_count:03}_preprocessed.jpg', preprocessed_frame)

        cap.release()

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
        # print("Video data shape:", video_data.shape)
        processed_videos.append((video_data, label, video_data.shape[0]))
        labels_list.append(label)

    X = np.array([video[0] for video in processed_videos])
    y = np.array([video[1] for video in processed_videos])
    num_frames = np.array([video[2] for video in processed_videos])

    X = X.astype(np.float32) / 255.0

    idx = np.random.randint(len(processed_videos))
    video_label = processed_videos[idx][1]
    video_path = video_paths_labels[idx]
    print(f"{video_path}")

    return X, y, num_frames, np.array(labels_list)

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
    data_resized = data_resized.reshape(num_videos, NUM_FRAMES, *target_shape, order='C')
    data_resized = data_resized[:num_samples].astype(np.float32)

    # Use the provided labels
    labels = np.concatenate([healthy_labels, ill_labels])
    labels = labels[:num_samples]

    return data_resized, labels