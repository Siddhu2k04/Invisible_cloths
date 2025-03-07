import cv2
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
time.sleep(3)  # Allow camera to adjust

# Capture background (ensure no object is present)
ret, background = cap.read()
if not ret:
    print("Error: Unable to capture background.")
    cap.release()
    exit()
background = cv2.flip(background, 1)  # Flip for alignment

# Define kernel for smoothing
kernel = np.ones((5, 5), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Flip frame for mirror effect

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define white color range in HSV
    lower_white = np.array([0, 0, 100])  # Adjust if needed
    upper_white = np.array([180, 50, 255])  

    # Create mask to detect white color
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Refine mask with Morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)  # Fill gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)   # Remove noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)  # Smooth edges

    # Invert mask to get non-white areas
    mask_inv = cv2.bitwise_not(mask)

    # Extract white regions from background
    background_part = cv2.bitwise_and(background, background, mask=mask)

    # Extract non-white parts from current frame
    frame_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Combine both results
    final_output = cv2.addWeighted(background_part, 1, frame_part, 1, 0)

    # Display output
    cv2.imshow("White to Invisible Cloak Effect", final_output)

    # Exit on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows() 