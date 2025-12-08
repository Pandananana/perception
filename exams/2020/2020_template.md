Here is the markdown document created from the PDF, including all available options for each question and using the requested checkbox format `- [ ]`.

# 📄 Exam 31392: Perception for Autonomous Systems (May 2020)

**Duration:** 4 hours
**Total Questions:** 34 (excluding Question 0)
**Note:** Each question has a different weight indicated in its description.

***

## Question 0 (0.0% of the exam)

Welcome to the Exam of the course 31392: Perception for Autonomous Systems!
Materials and datasets for all "numerical" exercises are available via a link.

* Do you understand that you can either download all the files now or one-by-one in each question?
  - [ ] Yes
  - [ ] No
* Do you understand that in both cases you will have access to the exact same files?
  - [ ] Yes
  - [ ] No

***

## Question 1 (2.0% of the test grade):

Consider the matrix below. Assume that we want to convolve an image with this matrix. What will be the outcome?

| 10 | 10 | 0 |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 0 |

**Options:**
- [ ] The image will be shifted downwards by 1 pixel
- [ ] The image will remain unaffected
- [ ] The image will be blurred
- [ ] The image will be sharpened
- [ ] The image will be shifted to the right by 1 pixel
- [ ] The image will be shifted upwards by 1 pixel

***

## Question 2 (2.0% of the test grade):

What is the output of a smoothing, linear spatial filter?

- [ ] Median of pixels
- [ ] Maximum of pixels
- [ ] Minimum of pixels
- [ ] Average of pixels

***

## Question 3 (2.0% of the test grade):

Which of the following filters response is based on ranking of pixel values?

- [ ] Nonlinear smoothing filters
- [ ] Linear Smoothing Filters
- [ ] Sharpening Filters

***

## Question 4 (2.0% of the test grade):

Consider that we are using the following kernel to perform convolution and correlation on an image. Do you expect the results of the two operations to be different or identical and why?

| 0.20 | 0 | 0.2 |
| :---: | :---: | :---: |
| 0 | 0.20 | 0 |
| 0.20 | 0 | 0.2 |

- [ ] The result will be different because none of the kernel elements is larger than 0.5
- [ ] The results will be identical because the kernel is symmetric
- [ ] The results will be identical because all the elements of the kernel add up to 1
- [ ] The results will be different because convolution and correlation are two different operations

***

## Question 5 (2.0% of the test grade):

Can we perform correlation using convolution?

- [ ] Yes, by tweaking the kernel
- [ ] Yes, by using different input image
- [ ] No, correlation and convolution are fundamentally different
- [ ] No, we can only perform convolution using correlation, but not the inverse

***

## Question 6 (2.0% of the test grade): Harris Detector

Select TRUE or False for each one of the following statements:

| Statement | True | False |
| :--- | :---: | :---: |
| The Harris detector is used in computer vision to detect **Corners** | - [ ] | - [ ] |
| The Harris detector is used in computer vision to detect **Edges** | - [ ] | - [ ] |
| Harris detector uses the eigenvalues of the **Hessian Matrix** of the Image | - [ ] | - [ ] |
| Assuming $\lambda_1$ and $\lambda_2$ the eigenvalues of the Hessian Matrix M, points where $\lambda_1 >> \lambda_2$ are considered **edges** | - [ ] | - [ ] |
| Assuming $\lambda_1$ and $\lambda_2$ the eigenvalues of the Hessian Matrix M, points where $\lambda_2 >> \lambda_1$ are considered **edges** | - [ ] | - [ ] |
| Assuming $\lambda_1$ and $\lambda_2$ the eigenvalues of the Hessian Matrix M, points where $\lambda_1 \approx \lambda_2$ and large are considered **corners** | - [ ] | - [ ] |

***

## Question 7 (2.0% of the test grade): SIFT algorithm

Select TRUE or False for each one of the following statements:

| Statement | True | False |
| :--- | :---: | :---: |
| The **detection** is based on the **Difference of Gaussians** | - [ ] | - [ ] |
| The detection is based on the **Gradient** of the Image | - [ ] | - [ ] |
| The **description** is based on the Difference of Gaussians | - [ ] | - [ ] |
| The **description** is based on the **Gradient** of the Image | - [ ] | - [ ] |
| The fingerprint of the SIFT algorithm is vector with **64** values | - [ ] | - [ ] |
| The fingerprint of the SIFT algorithm is vector with **128** values | - [ ] | - [ ] |

***

## Question 8 (2.0% of the test grade): Hough transform

Select TRUE or False for each one of the following statements:

| Statement | True | False |
| :--- | :---: | :---: |
| The Hough transform can be used to detect **lines** | - [ ] | - [ ] |
| The Hough transform can be used to detect **circles** | - [ ] | - [ ] |
| The Hough transform is a **model fitting** algorithm | - [ ] | - [ ] |
| The Hough transform is **only** used in Computer Vision | - [ ] | - [ ] |
| Hough transform **can't handle outliers** | - [ ] | - [ ] |
| RANSAC should be used as an alternative to the Hough transform when the **dimension of feature is high** | - [ ] | - [ ] |

***

## Question 9 (2.0% of the test grade) - Numerical Hough Line Detection

**Instructions:**
1. Convert the image to grayscale (do not change the size!).
2. Apply Canny edge detection (`cv2.Canny`) with parameters "threshold1 $=100$" and "threshold2 $=200$".
3. Apply Hough Lines (`cv2.HoughLines`) with parameters "rho $=1$", "theta $=0.0017$", "threshold $I=200$".

**Question:** How many lines are detected?

- [ ] 16
- [ ] 32
- [ ] 64
- [ ] 128

***

## Question 10 (2.0% of the test grade) - Numerical - Harris Corner Detection

**Instructions:**
1. Convert the image to grayscale (do not change the size!).
2. Apply Harris corner (`cv2.cornerHarris`) with parameters "blockSize $=2$", "ksize $=3$", $k=0.04$.

**Question:** From the result of Harris Corners, how many values are above 0.01?

- [ ] 128
- [ ] 158
- [ ] 188
- [ ] 208

***

## Question 11 (2.0% of the test grade) - Numerical Optical Flow

**Instructions:**
1. Load the two images and convert them to grayscale (do not change the size!).
2. Use `cv2.goodFeaturesToTrack` to find features on the first image using "maxCorners $=100$", "qualityLevel $=0.3$" and "minDistance $=7$".
3. Apply sparse optical flow using the function `cv2.calcOpticalFlowPyrLK()`.

**Question:** What is the maximum amount of pixels moved for any object in the x direction (horizontally)?

- [ ] $\sim 15.5$
- [ ] $\sim 12.5$
- [ ] $\sim 20.5$
- [ ] $\sim 22.5$

***

## Question 12 (4.0% of the test grade): SAD Template Matching

**Measured series:** $[47, 211, 38, 53, 204, 116, 152, 249, 143, 177]$
**Object shape (template):** $[39, 55, 207]$
**Method:** SAD template matching.

**Question:** What is the best matching triplet in your measurements with your known object shape and what is the SAD for this match?

| Triplet | SAD Score |
| :--- | :--- |
| $[47, 211, 38]$ | 333 |
| $[116, 152, 249]$ | 6 |
| $[38, 53, 204]$ | 2 |

***

## Question 13 (2.0% of the test grade): Rectified Stereo

Select Correct or Wrong for each one of the following statements about a rectified stereo system:

| Statement | Correct | Wrong |
| :--- | :---: | :---: |
| Epipolar geometry can be used to describe both **unrectified and rectified** stereo cases | - [ ] | - [ ] |
| The rectified stereo case is **computationally simpler** to treat | - [ ] | - [ ] |
| We can obtain rectified stereo image pairs in practice by **physically mounting** the two sensors on a common plane | - [ ] | - [ ] |
| **Epipoles do not exist** in rectified stereo, because all epipolar lines are parallel | - [ ] | - [ ] |
| The disparity value of a point **grows with its depth** | - [ ] | - [ ] |
| We cannot obtain depth from stereo without knowing the system's **focal length and baseline** | - [ ] | - [ ] |

***

## Question 14 (2.0% of the test grade): Stereo Correspondence Algorithms

Select Correct or Wrong for each one of the following statements:

| Statement | Correct | Wrong |
| :--- | :---: | :---: |
| Local algorithms typically produce **inferior disparity maps**, compared to global algorithms | - [ ] | - [ ] |
| **Bigger windows** for calculating dis-similarity metrics... result **always in better disparity results** | - [ ] | - [ ] |
| **Bigger windows** for calculating dis-similarity metrics... result **always in worse disparity results** | - [ ] | - [ ] |
| **Smaller windows are typically preferred** for images with finer and more complicated texture | - [ ] | - [ ] |

***

## Question 15 (2.0% of the test grade):

Could convolution be used to implement (dis-)similarity calculations in stereo matching?

- [ ] Yes, but with proper considerations for formulating the kernel
- [ ] No, it is correlation that expresses (dis-)similarity
- [ ] Yes, because in (dis-)similarity calculations the considered kernel is symmetric
- [ ] No, because in convolution the kernel needs to be smaller than the image

***

## Question 16 (2.0% of the test grade): Monocular Camera Projection and Calibration

Select TRUE or False for each one of the following statements:

| Statement | TRUE | FALSE |
| :--- | :---: | :---: |
| The projection matrix includes **instrinsic** parameters | - [ ] | - [ ] |
| The projection matrix includes **extrinsic** parameters | - [ ] | - [ ] |
| The **Homography** is used to project a point from **3D to 2D** | - [ ] | - [ ] |
| When using a **flat calibration pattern** we can employ the homography to perform calibration | - [ ] | - [ ] |
| Lens distortion is modeled as a **polynomial** | - [ ] | - [ ] |

***

## Question 17 (2.0% of the test grade): Stereo Camera Projection and Calibration

Select TRUE or False for each one of the following statements:

| Statement | TRUE | FALSE |
| :--- | :---: | :---: |
| The Fundamental Matrix describes **only** extrinsic parameters | - [ ] | - [ ] |
| The Essential matrix contains **only extrinsic** parameters | - [ ] | - [ ] |
| The camera matrix can be calculated from the **Fundamental Matrix** | - [ ] | - [ ] |
| The Fundamental matrix projects image points from one image of the stereo pair to the other | - [ ] | - [ ] |
| The epipoles of a stereo pair are found at the intersection of the **epipolar lines** | - [ ] | - [ ] |
| The epipoles of the stereo system are found at the intersection of the **baseline with the camera planes** | - [ ] | - [ ] |

***

## Question 18 (6.0% of the test grade): - Numerical - Stereo Calibration

**Question:** Determine the approximate coefficients of the epipolar lines in the imageset `left.png` and `right.png` by taking the average of all the epipolar lines. You should use 2000 of the best matching SIFT keypoints to compute the epipolar lines.
*(Note: OpenCV computes the epipolar lines in the form $ax+by+c=0$.)*

**Options:**
- [ ] $[-1.69333264 \mathrm{e}-02 \quad 7.97691572 \mathrm{e}-01 \quad -0.26727486 \mathrm{e}+02]$
- [ ] $[ 17.0291542 \mathrm{e}-02 \quad -7.9747336 \mathrm{e}-01 \quad 0.2541019 \mathrm{e}+02]$

***

## Question 19 (3.0% of the test grade): RANSAC Iterations

The number of iterations $k$ that RANSAC needs is defined by the formula: $$k = \frac{\log(1-p)}{\log(1-w^s)}$$
where $k$ is the number of iterations, $p$ is the probability of RANSAC having chosen a set of points free of outliers, and $w$ is the proportion of inliers with respect to all the points in the dataset (assuming $s=2$ for the minimum number of points).

### SUB-QUESTION A

Assume that we want a probability of success $p \ge 98\%$ and the proportion of inliers $w = 75\%$. How many iterations does RANSAC need to achieve this?

- [ ] The number of needed iterations is: $k=3$
- [ ] The number of needed iterations is: $k=5$
- [ ] The number of needed iterations is: $k=14$
- [ ] The number of needed iterations is: $k=64$

### SUB-QUESTION B

How would the number of needed iterations change if the size of our dataset (number of points) doubled, but all other aspects of our scenario remained the same?

- [ ] With a dataset twice as big, the number of required iterations would be half
- [ ] With a dataset twice as big, the number of required iterations would be the same
- [ ] With a dataset twice as big, the number of required iterations would be double

***

## Question 20 (3.0% of the test grade): Iterative Closest Point (ICP)

Select Correct or Wrong for each one of the following statements:

| Statement | Correct | Wrong |
| :--- | :---: | :---: |
| ICP is a deterministic algorithm | - [ ] | - [ ] |
| ICP is guaranteed to converge... no matter what the initial relative pose... is | - [ ] | - [ ] |
| ICP is one of the underlying algorithms for implementing the Kabsch algorithm | - [ ] | - [ ] |
| In ICP, all intermediate transformations until convergence are applied always to the same of the two point clouds | - [ ] | - [ ] |
| ICP could employ either Spin Images or FPFH for finding the intermediate transformations until convergence | - [ ] | - [ ] |
| ICP is not particularly robust to outliers | - [ ] | - [ ] |
| When using ICP, one needs to choose either the Kabsch algorithm or the Procrustes analysis | - [ ] | - [ ] |

***

## Question 21 (1.0% of the test grade): Point Cloud Registration

In Point Cloud Registration, local alignment usually takes place first, and global alignment second. Is the statement correct or wrong?

- [ ] Correct
- [ ] Wrong

***

## Question 22 (1.0% of the test grade): Kabsch Algorithm

The Kabsch algorithm is typically used both for Local and Global alignment.

- [ ] Yes, the Kabsch algorithm is used in both cases
- [ ] No, the Kabsch algorithm is only used for Local alignment
- [ ] No, the Kabsch algorithm is only used for Global alignment

***

## Question 23 (10.0% of the test grade): K-means and PCA

Apply K-means and PCA on a provided dataset.

### SUB-QUESTION A (K-means Optimal K)

Use the elbow method to find how many clusters are optimal for this dataset.

- [ ] The optimal K is: 3
- [ ] The optimal K is: 4
- [ ] The optimal K is: 5
- [ ] The optimal K is: 6

### SUB-QUESTION B (PCA Components)

Perform PCA and determine the minimum number of components required to express 95% of the variance.

- [ ] For 95%, the number of needed components is: 3
- [ ] For 95%, the number of needed components is: 4
- [ ] For 95%, the number of needed components is: 5
- [ ] For 95%, the number of needed components is: 6

### SUB-QUESTION C (K-means Inertia)

Perform K-means on the PCA transformed data using the optimal K from SUB-QUESTION A and the PCA transformed data from SUB-QUESTION B. Calculate the Inertia (sum of squared differences of samples to the closest centroid).

- [ ] Inertia has a value between: $3000$ and $3500$
- [ ] Inertia has a value between: $4000$ and $5000$
- [ ] Inertia has a value between: $5000$ and $5500$

***

## Question 24 (6.0% of the test grade): Linear Regression

Apply Linear Regression to the provided data to obtain a model of the form $y=ax+b$.

**Question:** What are the values of the parameters "$a$" and "$b$" in that model (APPROXIMATELY)?

| Parameter | Value |
| :--- | :--- |
| $a$ | $0.55$ |
| $a$ | $0.23$ |
| $a$ | $0.82$ |
| $b$ | $12.37$ |
| $b$ | $19.86$ |
| $b$ | $25.60$ |

***

## Question 25 (1.0% of the test grade): DBSCAN

The clustering algorithm DBSCAN needs to assign some points as "noise points". Is this statement correct or wrong?

- [ ] Correct
- [ ] Wrong

***

## Question 26 (2.5% of the test grade): State Estimation

Select TRUE or False for each one of the following statements:

| Statement | TRUE | FALSE |
| :--- | :---: | :---: |
| Histogram Filter Concerns **Discrete States** | - [ ] | - [ ] |
| Kalman Filter Concerns **Discrete States** | - [ ] | - [ ] |
| Histogram Filter Concerns **Unimodal Uncertainty** Distributions | - [ ] | - [ ] |
| Kalman Filter Concerns **Unimodal Uncertainty** Distributions | - [ ] | - [ ] |
| In the Histogram filter, measurement involves **Bayes rule** and movement involves **convolution** | - [ ] | - [ ] |
| In Kalman filter, the variance of the estimation is **higher after measurement** | - [ ] | - [ ] |
| In Kalman filter, the variance of the estimation is **higher after movement** | - [ ] | - [ ] |

***

## Question 27 (5.0% of the test grade) - Numerical - Kalman Filter

**Current state ($\mathbf{x}_k$):**
$$\mathbf{x}_k = \begin{bmatrix} x \\ x' \\ y \\ y' \end{bmatrix} = \begin{bmatrix} 3 \\ 0.5 \\ 2 \\ 0.33 \end{bmatrix}$$

**Current covariance ($\mathbf{P}_k$):**
$$\mathbf{P}_k = \begin{bmatrix} 5 & 1 & 0 & 0 \\ 1 & 2 & 0 & 0 \\ 0 & 0 & 5 & 1 \\ 0 & 0 & 1 & 2 \end{bmatrix}$$

**Assumptions:** Constant velocity model, no external forces ($\mathbf{u}=0$), no process noise ($\mathbf{Q}=0$), timestep $dt=1$.

**Question:** What is the next predicted state ($\mathbf{x}_{k+1}$) and the next predicted covariance ($\mathbf{P}_{k+1}$)?

**Options (Predicted State $\mathbf{x}_{k+1}$):**
- [ ] $$\mathbf{x}_{k+1} = \begin{bmatrix} 3.5 \\ 0.5 \\ 2.33 \\ 0.33 \end{bmatrix}$$

**Options (Predicted Covariance $\mathbf{P}_{k+1}$):**
- [ ] $$\mathbf{P}_{k+1} = \begin{bmatrix} 9 & 3 & 0 & 0 \\ 3 & 2 & 0 & 0 \\ 0 & 0 & 9 & 3 \\ 0 & 0 & 3 & 2 \end{bmatrix}$$

***

## Question 28 (2.5% of the test grade) - Numerical - Kalman Tracking

**Predicted state ($\mathbf{x}_k^-$):**
$$\mathbf{x}_k^- = \begin{bmatrix} x \\ x' \\ y \\ y' \end{bmatrix} = \begin{bmatrix} 5 \\ 0.5 \\ 7 \\ 0.8 \end{bmatrix}$$

**Predicted covariance ($\mathbf{P}_k^-$):**
$$\mathbf{P}_k^- = \begin{bmatrix} 0.2 & 0 & 0 & 0 \\ 0.2 & 0.1 & 0 & 0 \\ 0 & 0 & 0.2 & 0 \\ 0 & 0 & 0.2 & 0.1 \end{bmatrix}$$

**Measurement ($\mathbf{z}_k$):**
$$\mathbf{z}_k = \begin{bmatrix} 4.8 \\ 7.1 \end{bmatrix}$$

**Observation noise ($\mathbf{R}$):**
$$\mathbf{R} = \begin{bmatrix} 0.2 & 0.2 \\ 0.2 & 0.2 \end{bmatrix}$$

**Question:** What is the state after having updated it with the current measurement (do not run the predict step, only the update step)?

**Option (Updated State $\mathbf{x}_k^+$):**
- [ ] $$\mathbf{x}_k^+ = \begin{bmatrix} 4.833333 \\ 0.333333 \\ 7.133333 \\ 0.933333 \end{bmatrix}$$

***

## Question 29 (6.0% of the test grade): Support Vector Machine (SVM)

You will train and test a Support Vector Machine (SVM).

### SUB-QUESTION A

Examine the provided code. What proportion of the original dataset is used for training?

- [ ] Proportion used for training: $30\%$
- [ ] Proportion used for training: $70\%$
- [ ] Proportion used for training: $75\%$

### SUB-QUESTION B

Apply a SVM as instructed in the provided code. What is the reported accuracy of your SVM (APPROXIMATELY)?

- [ ] Accuracy: $0.83$
- [ ] Accuracy: $0.92$
- [ ] Accuracy: $0.96$

***

## Question 30 (1.0% of the test grade): k-NN

The classification algorithm k-NN requires approximately similar time for training and testing (inference). Is this statement correct or wrong?

- [ ] Correct
- [ ] Wrong

***

## Question 31 (2.0% of the test grade): Visual Odometry

Select TRUE or False for each one of the following statements:

| Statement | TRUE | FALSE |
| :--- | :---: | :---: |
| The 3D to 3D methods are **less accurate** than the 3D to 2D ones | - [ ] | - [ ] |
| The axis angle representation suffers from the **"gimbal lock" problem** | - [ ] | - [ ] |
| Visual Odometry can **only be performed on a frame to frame manner** | - [ ] | - [ ] |
| 3D to 3D approaches use the **image coordinates** of the current frame and the **3D position** of the previous frame to estimate the motion | - [ ] | - [ ] |
| 2D to 2D methods are **"accurate up to scale"** | - [ ] | - [ ] |
| 3D to 2D methods use the **PnP solutions** | - [ ] | - [ ] |
| 2D to 2D methods are equivalent to calculating the **essential matrix** | - [ ] | - [ ] |

***

## Question 32 (10.0% of the test grade) - Numerical - Visual Odometry

**Goal:** Compute the final Pose (Position: [X,Y,Z] Orientation: [axis angles x, axis angles y, axis angles z]) of the car.

**Steps:**
* **Do only once:** Capture $I_{k-2}, I_{k-1}$. Extract and match features. Triangulate features from $I_{k-2}, I_{k-1}$.
* **Do at each iteration:** Capture new frame $I_k$. Extract and match features with $I_{k-1}$. Compute camera pose (PnP) from 3-D-to-2-D matches. Triangulate all new feature matches between $I_k$ and $I_{k-1}$.

**Options:**
- [ ] Position = $[-11.908] [0.984] [1.727]$, Rotation = $[0.000] [1.296] [0.002]$
- [ ] Position = $[-16.908] [0.684] [11.727]$, Rotation = $[0.0102] [1.596] [0.072]$
- [ ] Position = $[-6.908] [0.384] [4.727]$, Rotation = $[0.040] [1.396] [0.271]$
- [ ] Position = $[-7.908] [0.184] [7.727]$, Rotation = $[0.040] [1.996] [0.471]$

***

## Question 33 (2.0% of the test grade): Non-linear Least Squares Optimization for SLAM

Select TRUE or False for each one of the following statements:

| Statement | TRUE | FALSE |
| :--- | :---: | :---: |
| Gauss-Newton suffers from **overshooting** | - [ ] | - [ ] |
| Gradient-Descent suffers from **slow convergence** | - [ ] | - [ ] |
| Levenberg Marquadt is a **combination** of Gauss Newton and Gradient Descent | - [ ] | - [ ] |
| Levenberg Marquadt suffers from **all the above** | - [ ] | - [ ] |

***

## Question 34 (2.0% of the test grade): SLAM

Select TRUE or False for each one of the following statements:

| Statement | True | False |
| :--- | :---: | :---: |
| SLAM is concerned **solely** with the identification of the position of a camera | - [ ] | - [ ] |
| SLAM is defined as a **graph where nodes are the poses/landmarks** and **edges are the observations/odometry** | - [ ] | - [ ] |
| The goal of SLAM is to **satisfy all the constrains in the pose graph** | - [ ] | - [ ] |