import cv2
import numpy as np

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