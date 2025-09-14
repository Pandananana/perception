import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Performance settings
SCALE_FACTOR = 0.5  # Process at half resolution for speed
TARGET_FPS = 30
SKIP_FRAMES_THRESHOLD = 0.1  # Skip frames if processing takes too long

video = cv2.VideoCapture("Robots.mp4")
frame_buffer = []
frame_count = 0

while True:
    ret, frame = video.read()
    frame_count += 1
    frame_buffer.append(frame) if frame_count % 2 == 0 else None

    if not ret:
        break

    if len(frame_buffer) < 2:
        cv2.imshow("Frame", frame)
        continue

    # Scale down frames for faster processing
    h, w = frame.shape[:2]
    new_h, new_w = int(h * SCALE_FACTOR), int(w * SCALE_FACTOR)

    frame1_small = cv2.resize(frame_buffer[0], (new_w, new_h))
    frame2_small = cv2.resize(frame, (new_w, new_h))

    gray1 = cv2.cvtColor(frame1_small, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_small, cv2.COLOR_BGR2GRAY)

    # Optimized Farneback parameters for speed
    # Parameters: pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    # Reduced iterations and smaller window size for speed
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 2, 7, 2, 5, 1.1, 0)

    mag, ang = cv2.cartToPolar(
        flow[:, :, 0], flow[:, :, 1]
    )  # Magnitude and angle of flow

    # Sample the flow at regular intervals to avoid clutter
    step = 20  # Sample every 20 pixels
    h, w = flow.shape[:2]

    # Create a grid of points to sample
    y, x = (
        np.mgrid[step // 2 : h : step, step // 2 : w : step].reshape(2, -1).astype(int)
    )

    # Draw arrows for sampled points
    for i in range(len(x)):
        px, py = x[i], y[i]

        # Get flow vector at this point
        fx, fy = flow[py, px]

        # Only draw arrow if flow magnitude is significant
        if mag[py, px] > 1.0:  # Threshold to filter out small movements
            # Calculate end point of arrow
            end_x = int(px + fx)
            end_y = int(py + fy)

            # Draw arrow using cv2.arrowedLine
            cv2.arrowedLine(
                frame2_small, (px, py), (end_x, end_y), (0, 255, 0), 1, tipLength=0.3
            )

    cv2.imshow("Frame", frame2_small)

    if len(frame_buffer) > 2:
        frame_buffer.pop(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
