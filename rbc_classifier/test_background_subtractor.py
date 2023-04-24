import cv2
import numpy as np
from background_subtractor import BackgroundSubtractor

# Path to the video file
video_path = '/data/RBC_Phantom_60xOlympus/Donor_1/Native5_focused/1310_03311.avi'

# Create a background subtractor object
bg_subtractor = BackgroundSubtractor()

# Open the video file
cap = cv2.VideoCapture(video_path, cv2.CAP_ANY)

# Check if the video file was successfully opened
if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction to the frame
    fg = bg_subtractor.apply(frame)

    # Display the original frame and the foreground mask
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground mask', fg)

    # Wait for a key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
