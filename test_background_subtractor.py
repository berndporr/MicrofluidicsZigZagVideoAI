import os
import cv2
import logging

video_path = '/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused/1310_03454.avi'

# Create the save directory path
save_directory = os.path.join(os.getcwd(), 'frames')

# Create the "frames" folder if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

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

        # Skip the first and last 10 frames.
        if 50 <= frame_count < 150:
            # Skip every other frame.
            if frame_count % 10 == 0:
                video_frames.append(processed_frame)

                # Save the frame.
                cv2.imwrite(f'{save_directory}/{frame_count:03}_original.jpg', frame)
                cv2.imwrite(f'{save_directory}/{frame_count:03}_processed.jpg', processed_frame)
                # cv2.imwrite(f'{save_directory}/{frame_count:03}_fg_mask.png', fgMask)

        # # Display processed frame.
        # cv2.imshow('Processed video', processed_frame)

    # Close the video file.
    cap.release()

# # Wait for keypress and exit.
# cv2.waitKey(0)
# cv2.destroyAllWindows()

