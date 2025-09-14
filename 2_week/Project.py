import cv2
import numpy as np
import matplotlib.pyplot as plt

video = cv2.VideoCapture("Robots.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
