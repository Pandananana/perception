import cv2
import numpy as np
import matplotlib.pyplot as plt

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

    gray1 = cv2.cvtColor(frame_buffer[0], cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame_buffer[1], cv2.COLOR_BGR2GRAY)

    feat1 = cv2.goodFeaturesToTrack(
        gray1, maxCorners=100, qualityLevel=0.3, minDistance=7
    )
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)

    for i in range(len(feat1)):
        f10 = int(feat1[i][0][0])
        f11 = int(feat1[i][0][1])
        f20 = int(feat2[i][0][0])
        f21 = int(feat2[i][0][1])

        # Calculate the distance between the two points
        dist = np.sqrt((f10 - f20) ** 2 + (f11 - f21) ** 2)

        # Only draw the line if the distance is greater than 10
        if dist > 2.0:
            cv2.line(frame, (f10, f11), (f20, f21), (0, 255, 0), 2)
            cv2.circle(frame, (f10, f11), 5, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)

    if len(frame_buffer) > 2:
        frame_buffer.pop(0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
