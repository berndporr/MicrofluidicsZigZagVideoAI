import cv2
import logging

video_path = '/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused/1310_03311.avi'

video = [video_path]

bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)

for video_path in video:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.warning(f"Could not open video file: {video_path}")
        exit()

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

        # Display processed frame.
        cv2.imshow('Processed video', processed_frame)

    # Close the video file.
    cap.release()

# Wait for keypress and exit.
cv2.waitKey(0)
cv2.destroyAllWindows()

