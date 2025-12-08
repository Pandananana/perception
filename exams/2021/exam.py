# %%
import numpy as np
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import math

from clustering import plot_elbow_graph

# %% ######################################################
# Question 10: Hough Lines
# #########################################################
image = cv2.imread("data/books.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)
lines = cv2.HoughLines(edges, rho=1, theta=0.0017, threshold=200)

print(f"Number of Hough Lines: {len(lines)}")
# %% ######################################################
# Question 11: Harris Corners
# #########################################################
image = cv2.imread("data/books.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

print(f"Number of Harris Corners over threshold: {np.sum(corners > 0.01)}")
# %% ######################################################
# Question 12: Optical Flow
# #########################################################
image1 = cv2.imread("data/things1.png")
image2 = cv2.imread("data/things2.png")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

features = cv2.goodFeaturesToTrack(
    gray1, maxCorners=100, qualityLevel=0.3, minDistance=7
)

features2, status, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None)

largest_displacements = np.max(abs(features - features2), axis=0)

print(f"Largest displacement in x: {largest_displacements[0, 0]:.1f}")

# %% ######################################################
# Question 18: Epipolar Lines
# #########################################################
left = cv2.imread("data/left.png", cv2.IMREAD_GRAYSCALE)
right = cv2.imread("data/right.png", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT.create()

keypoints_left, descriptors_left = sift.detectAndCompute(left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right, None)

brute_force_matcher = cv2.BFMatcher()
matches = brute_force_matcher.match(descriptors_left, descriptors_right)

matches = sorted(matches, key=lambda x: x.distance)

num_best_matches = 2000

points1 = []
points2 = []

for m in matches[:num_best_matches]:
    points1.append(keypoints_left[m.queryIdx].pt)
    points2.append(keypoints_right[m.trainIdx].pt)

points1 = np.int32(points1)
points2 = np.int32(points2)

F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

# We select only inlier points
points1 = points1[mask.ravel() == 1]
points2 = points2[mask.ravel() == 1]

lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)

average_epipolar_line1 = np.average(lines1, axis=0)
print(f"Average epipolar line 1: {average_epipolar_line1}")


lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
average_epipolar_line2 = np.average(lines2, axis=0)

print(f"Average epipolar line 2: {average_epipolar_line2}")

# %% ######################################################
# Question 19: RANSAC probability
# #########################################################
p = 0.98
w = 0.75
k = math.log(1 - p) / math.log(1 - w**2)
print(f"k = {k:.0f}")

# %% ######################################################
# Question 23: KMeans and PCA
# #########################################################
with open("data/clusters.txt", "r") as f:
    data = np.loadtxt(f, dtype=int)


# %% Question 23.1: Elbow Method
plot_elbow_graph(data, 2, 10)  ## 5 clusters

# %% Question 23.2: PCA
pca = PCA(0.95)

# scaler = StandardScaler()
# scaler.fit(data)

# scaled_data = scaler.transform(data)

pca.fit(data)

print(f"Number of components: {pca.n_components_}")

# %% Question 23.3: KMeans on PCA
pca_data = pca.transform(data)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(pca_data)

distortion = 0

for pca_point in pca_data:
    cluster_index = kmeans.predict([pca_point])
    pca_cluster_center = kmeans.cluster_centers_[cluster_index]

    cluster_center = pca.inverse_transform(pca_cluster_center)
    # cluster_center = scaler.inverse_transform(scaled_cluster_center)

    point = pca.inverse_transform([pca_point])
    # point = scaler.inverse_transform(scaled_point)

    distortion += (pca_point - pca_cluster_center) ** 2

print(f"Distortion: {distortion.sum()}")
print(f"Inertia: {kmeans.inertia_}")

# %% ######################################################
# Question 24: Linear Regression
# #########################################################
with open("data/lr_x.txt", "r") as f:
    x = np.loadtxt(f)

with open("data/lr_y.txt", "r") as f:
    y = np.loadtxt(f)

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

regression = LinearRegression()
regression.fit(x, y)

print(f"Coefficient (a): {regression.coef_[0, 0]:.2f}")
print(f"Intercept (b): {regression.intercept_[0]:.2f}")

# %% ######################################################
# Question 27: Kalman filter State Transition
# #########################################################
x = 3
dx = 0.5
y = 2
dy = 0.33

state = np.array([x, dx, y, dy])

covariance = [
    [5, 1, 0, 0],
    [1, 2, 0, 0],
    [0, 0, 5, 1],
    [0, 0, 1, 2],
]

dt = 1

state_transition = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

next_state = state_transition @ state

print(f"Next state: {next_state}")

next_covariance = state_transition @ covariance @ state_transition.T
print(f"Next covariance:\n {next_covariance}")


# %% ######################################################
# Question 28: Kalman filter
# #########################################################

x = 5
dx = 0.5
y = 7
dy = 0.8

state = np.array([x, dx, y, dy])  # x

covariance = [
    [0.2, 0, 0, 0],
    [0.2, 0.1, 0, 0],
    [0, 0, 0.2, 0],
    [0, 0, 0.2, 0.1],
]  # P

measurement = np.array([4.8, 7.1])  # z

observation_noise = 0.2 * np.ones((2, 2)) # R

observation_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # H


y = measurement - np.dot(observation_matrix, state)
S = (
    np.dot(observation_matrix, np.dot(covariance, observation_matrix.T))
    + observation_noise
)
K = np.dot(covariance, np.dot(observation_matrix.T, np.linalg.pinv(S)))
xd = state + np.dot(K, y)

print(xd)

Pd = np.dot((np.eye(4) - np.dot(K, observation_matrix)), covariance)
