"""
CameraPose Class for Computer Vision Exam Questions

Handles coordinate transformations between world and camera frames,
various input formats (PnP output, extrinsic matrices, etc.), and
common geometric computations.

Key Formulas:
    P_cam = R @ (P_world - C)     -- world to camera
    P_cam = R @ P_world + t       -- alternative form
    P_world = R^T @ P_cam + C     -- camera to world
    t = -R @ C                    -- translation from camera center
    C = -R^T @ t                  -- camera center from translation

Author: Andreas (DTU Autonomous Systems)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Tuple, Optional


class CameraPose:
    """
    Camera pose representation for computer vision problems.

    Coordinate Systems:
        - World frame: Fixed reference frame (e.g., checkerboard origin)
        - Camera frame: Origin at camera center, Z forward, X right, Y down

    Usage:
        # From PnP solver output (most common exam case)
        pose = CameraPose.from_rvec_tvec(rvec, tvec)

        # Transform stereo reconstruction result to world
        P_world = pose.camera_to_world(P_cam)
    """

    def __init__(self, rotation: np.ndarray, position: np.ndarray):
        """
        Initialize camera pose.

        Args:
            rotation: 3x3 rotation matrix (world to camera rotation)
            position: Camera center in world coordinates (3,)
        """
        self.R = np.asarray(rotation)  # 3x3 rotation matrix (world-to-camera)
        self.C = np.asarray(position).flatten()  # Camera center in world coords

    # =========================================================================
    # FACTORY METHODS - Different ways exam might give you the pose
    # =========================================================================

    @classmethod
    def from_rvec_tvec(cls, rvec: np.ndarray, tvec: np.ndarray) -> "CameraPose":
        """
        Create from PnP solver output (OpenCV convention).

        This is what cv2.solvePnP() returns!

        Args:
            rvec: Rodrigues rotation vector (3,) - rotation axis * angle
            tvec: Translation vector (3,) - where origin is in camera frame

        Note: PnP satisfies P_cam = R @ P_world + t
        """
        rvec = np.asarray(rvec).flatten()
        tvec = np.asarray(tvec).flatten()

        # Rodrigues formula: convert rotation vector to matrix
        R = Rotation.from_rotvec(rvec).as_matrix()

        # Camera center: C = -R^T @ t
        # (solving for C in: t = -R @ C)
        C = -R.T @ tvec

        return cls(R, C)

    @classmethod
    def from_Rt(cls, R: np.ndarray, t: np.ndarray) -> "CameraPose":
        """
        Create from rotation matrix R and translation vector t.

        Where: P_cam = R @ P_world + t

        Args:
            R: 3x3 rotation matrix
            t: 3x1 or (3,) translation vector
        """
        R = np.asarray(R)
        t = np.asarray(t).flatten()
        C = -R.T @ t
        return cls(R, C)

    @classmethod
    def from_extrinsic_matrix(cls, E: np.ndarray) -> "CameraPose":
        """
        Create from extrinsic matrix.

        Accepts either:
            - 3x4 matrix [R | t]
            - 4x4 matrix [R | t; 0 0 0 1]

        Common exam format for transformation matrices.
        """
        E = np.asarray(E)
        R = E[:3, :3]
        t = E[:3, 3]
        C = -R.T @ t
        return cls(R, C)

    @classmethod
    def from_RC(cls, R: np.ndarray, C: np.ndarray) -> "CameraPose":
        """
        Create directly from R and camera center C.

        Use when exam gives camera center directly (rare but possible).

        Args:
            R: 3x3 rotation matrix
            C: Camera center in world coordinates
        """
        return cls(R, C)

    @classmethod
    def from_projection_matrix(
        cls, P: np.ndarray, K: Optional[np.ndarray] = None
    ) -> "CameraPose":
        """
        Create from 3x4 projection matrix P = K @ [R | t].

        Args:
            P: 3x4 projection matrix
            K: Optional 3x3 intrinsic matrix. If provided, uses direct computation.
               If not provided, uses RQ decomposition to extract K.

        Note: If you know K, provide it for more accurate results.
        """
        P = np.asarray(P)

        if K is not None:
            # Direct method: [R|t] = K^(-1) @ P
            K_inv = np.linalg.inv(K)
            Rt = K_inv @ P
            R = Rt[:, :3]
            t = Rt[:, 3]
        else:
            # RQ decomposition to extract K and R
            M = P[:, :3]

            # QR on flipped matrix, then flip back
            Q, R_upper = np.linalg.qr(np.flipud(M).T)
            R = np.flipud(Q.T)
            K_extracted = np.flipud(np.fliplr(R_upper.T))

            # Ensure positive diagonal for K
            D = np.diag(np.sign(np.diag(K_extracted)))
            K_extracted = K_extracted @ D
            R = D @ R

            # Ensure proper rotation (det = 1)
            if np.linalg.det(R) < 0:
                R = -R

            t = np.linalg.inv(K_extracted) @ P[:, 3]

        C = -R.T @ t
        return cls(R, C)

    # =========================================================================
    # COORDINATE TRANSFORMATIONS - Core exam operations
    # =========================================================================

    def world_to_camera(self, P_world: np.ndarray) -> np.ndarray:
        """
        Transform point(s) from world frame to camera frame.

        Formula: P_cam = R @ (P_world - C)

        Use when: You have a world point and need camera coordinates.

        Args:
            P_world: Point(s) in world coords, shape (3,) or (N, 3)
        Returns:
            Point(s) in camera coordinates, same shape as input
        """
        P_world = np.asarray(P_world)
        if P_world.ndim == 1:
            return self.R @ (P_world - self.C)
        else:
            # Batch processing: (N, 3) array
            return (self.R @ (P_world - self.C).T).T

    def camera_to_world(self, P_cam: np.ndarray) -> np.ndarray:
        """
        Transform point(s) from camera frame to world frame.

        Formula: P_world = R^T @ P_cam + C

        Use when:
            - Stereo reconstruction gives you points in camera coords
            - You need to find where something is in the world

        Args:
            P_cam: Point(s) in camera coords, shape (3,) or (N, 3)
        Returns:
            Point(s) in world coordinates, same shape as input
        """
        P_cam = np.asarray(P_cam)
        if P_cam.ndim == 1:
            return self.R.T @ P_cam + self.C
        else:
            return (self.R.T @ P_cam.T).T + self.C

    def project_point(self, P_world: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Project 3D world point to 2D image pixel coordinates.

        Formula:
            P_cam = R @ P_world + t
            p_homogeneous = K @ P_cam
            pixel = p_homogeneous[:2] / p_homogeneous[2]

        Args:
            P_world: 3D point in world coordinates (3,)
            K: 3x3 camera intrinsic matrix
        Returns:
            2D pixel coordinates (u, v)
        """
        P_cam = self.world_to_camera(P_world)
        p_homogeneous = K @ P_cam
        return p_homogeneous[:2] / p_homogeneous[2]

    def backproject_pixel(
        self, pixel: np.ndarray, depth: float, K: np.ndarray
    ) -> np.ndarray:
        """
        Backproject 2D pixel to 3D world point given depth.

        Args:
            pixel: 2D image coordinates (u, v)
            depth: Depth value (Z in camera frame)
            K: 3x3 intrinsic matrix
        Returns:
            3D point in world coordinates
        """
        pixel = np.asarray(pixel)
        K_inv = np.linalg.inv(K)

        # Pixel to normalized camera coordinates
        p_homogeneous = np.array([pixel[0], pixel[1], 1.0])
        p_normalized = K_inv @ p_homogeneous

        # Scale by depth to get 3D camera point
        P_cam = p_normalized * depth

        # Transform to world
        return self.camera_to_world(P_cam)

    # =========================================================================
    # GETTER METHODS - Different representations exam might ask for
    # =========================================================================

    @property
    def t(self) -> np.ndarray:
        """
        Translation vector where P_cam = R @ P_world + t.

        Formula: t = -R @ C

        This is what appears in the extrinsic matrix [R|t].
        """
        return -self.R @ self.C

    @property
    def rvec(self) -> np.ndarray:
        """
        Rodrigues rotation vector (OpenCV format).

        The vector direction is the rotation axis.
        The vector magnitude is the rotation angle (radians).

        Useful for comparing with cv2.solvePnP output.
        """
        return Rotation.from_matrix(self.R).as_rotvec()

    @property
    def tvec(self) -> np.ndarray:
        """Alias for t - matches OpenCV naming convention."""
        return self.t

    @property
    def euler_angles(self) -> np.ndarray:
        """
        Euler angles (roll, pitch, yaw) in radians.

        Convention: XYZ intrinsic rotations
            - Roll: rotation about X
            - Pitch: rotation about Y
            - Yaw: rotation about Z
        """
        return Rotation.from_matrix(self.R).as_euler("xyz")

    @property
    def euler_angles_deg(self) -> np.ndarray:
        """Euler angles in degrees - more human readable."""
        return np.degrees(self.euler_angles)

    @property
    def rotation_angle(self) -> float:
        """
        Total rotation angle in radians.

        This is the magnitude of the axis-angle representation.
        """
        return np.linalg.norm(self.rvec)

    @property
    def rotation_angle_deg(self) -> float:
        """Total rotation angle in degrees."""
        return np.degrees(self.rotation_angle)

    @property
    def rotation_axis(self) -> np.ndarray:
        """
        Rotation axis (unit vector).

        The camera is rotated by rotation_angle about this axis.
        """
        angle = self.rotation_angle
        if angle < 1e-10:
            return np.array([0, 0, 1])  # Arbitrary when no rotation
        return self.rvec / angle

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """
        4x4 extrinsic transformation matrix.

        Returns:
            [R | t]
            [0 | 1]

        This transforms homogeneous world coords to camera coords.
        """
        E = np.eye(4)
        E[:3, :3] = self.R
        E[:3, 3] = self.t
        return E

    @property
    def extrinsic_matrix_3x4(self) -> np.ndarray:
        """
        3x4 extrinsic matrix [R | t].

        Common format in exam questions and OpenCV functions.
        """
        return np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def extrinsic_matrix_inv(self) -> np.ndarray:
        """
        4x4 inverse extrinsic matrix (camera to world transform).

        Returns:
            [R^T | C]
            [0   | 1]
        """
        E_inv = np.eye(4)
        E_inv[:3, :3] = self.R.T
        E_inv[:3, 3] = self.C
        return E_inv

    def get_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        Get 3x4 projection matrix P = K @ [R | t].

        Args:
            K: 3x3 intrinsic matrix
        Returns:
            3x4 projection matrix

        Use this to project world points: p = P @ [X, Y, Z, 1]^T
        """
        return K @ self.extrinsic_matrix_3x4

    # =========================================================================
    # GEOMETRIC PROPERTIES - Common exam questions about camera geometry
    # =========================================================================

    def get_optical_axis(self) -> np.ndarray:
        """
        Camera viewing direction in world coordinates.

        The optical axis is the Z-axis of the camera frame,
        expressed in world coordinates.

        Returns:
            Unit vector pointing where camera looks (world frame)
        """
        # Z-axis of camera is [0, 0, 1] in camera frame
        # In world frame: R^T @ [0, 0, 1] = third row of R
        return self.R[2, :].copy()

    def get_camera_up(self) -> np.ndarray:
        """
        Camera "up" direction in world coordinates.

        Note: In standard camera convention, Y points DOWN in image,
        so this returns the negative Y axis direction.

        Returns:
            Unit vector pointing "up" from camera's perspective (world frame)
        """
        # Y-axis points down in camera frame, so up is -Y
        return -self.R[1, :].copy()

    def get_camera_right(self) -> np.ndarray:
        """
        Camera "right" direction in world coordinates.

        Returns:
            Unit vector pointing right from camera's perspective (world frame)
        """
        return self.R[0, :].copy()

    def get_camera_axes_in_world(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get camera X, Y, Z axes expressed in world coordinates.

        Returns:
            (X_axis, Y_axis, Z_axis) as unit vectors in world frame

        Camera convention: X=right, Y=down, Z=forward
        """
        X_axis = self.R[0, :].copy()  # Camera X (right) in world
        Y_axis = self.R[1, :].copy()  # Camera Y (down) in world
        Z_axis = self.R[2, :].copy()  # Camera Z (forward) in world
        return X_axis, Y_axis, Z_axis

    def distance_to_point(self, P_world: np.ndarray) -> float:
        """
        Euclidean distance from camera center to a world point.

        Args:
            P_world: Point in world coordinates
        Returns:
            Distance (always positive)
        """
        return float(np.linalg.norm(P_world - self.C))

    def depth_of_point(self, P_world: np.ndarray) -> float:
        """
        Depth of world point (Z coordinate in camera frame).

        Depth != Distance!
        - Depth: projection onto optical axis
        - Distance: straight-line Euclidean distance

        Depth is what stereo reconstruction computes.

        Args:
            P_world: Point in world coordinates
        Returns:
            Depth value (can be negative if behind camera)
        """
        P_cam = self.world_to_camera(P_world)
        return float(P_cam[2])

    def is_point_visible(
        self,
        P_world: np.ndarray,
        K: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """
        Check if a world point is visible to the camera.

        Args:
            P_world: Point in world coordinates
            K: Optional intrinsic matrix (needed for image bounds check)
            image_size: Optional (width, height) for bounds check

        Returns:
            True if point is in front of camera (and optionally in image bounds)
        """
        P_cam = self.world_to_camera(P_world)

        # Must be in front of camera
        if P_cam[2] <= 0:
            return False

        # If K and image size provided, check if in image bounds
        if K is not None and image_size is not None:
            pixel = self.project_point(P_world, K)
            w, h = image_size
            if pixel[0] < 0 or pixel[0] >= w or pixel[1] < 0 or pixel[1] >= h:
                return False

        return True

    # =========================================================================
    # RELATIVE POSE - Multi-camera / stereo questions
    # =========================================================================

    def relative_pose_to(self, other: "CameraPose") -> "CameraPose":
        """
        Pose of 'other' camera relative to this camera.

        Returns transformation from other's frame to this frame.
        Useful for stereo geometry calculations.

        Args:
            other: Another CameraPose (e.g., right camera in stereo pair)
        Returns:
            Relative pose
        """
        # R_rel rotates from other's frame to this frame
        R_rel = self.R @ other.R.T

        # Translation: other's center in this camera's frame
        t_rel = self.world_to_camera(other.C)

        return CameraPose.from_Rt(R_rel, t_rel)

    def baseline_to(self, other: "CameraPose") -> float:
        """
        Baseline distance to another camera.

        Args:
            other: Another CameraPose
        Returns:
            Euclidean distance between camera centers
        """
        return float(np.linalg.norm(self.C - other.C))

    def baseline_vector_to(self, other: "CameraPose") -> np.ndarray:
        """
        Baseline vector from this camera to another (world coords).

        Args:
            other: Another CameraPose
        Returns:
            Vector from this camera center to other camera center
        """
        return other.C - self.C

    # =========================================================================
    # ESSENTIAL/FUNDAMENTAL MATRIX - Epipolar geometry
    # =========================================================================

    def essential_matrix_to(self, other: "CameraPose") -> np.ndarray:
        """
        Compute essential matrix E from this camera to another.

        Satisfies: p2^T @ E @ p1 = 0 for corresponding normalized points

        Args:
            other: The second camera pose
        Returns:
            3x3 essential matrix
        """
        # Relative pose from camera 1 to camera 2
        R_rel = other.R @ self.R.T  # Rotation from cam1 to cam2
        t_rel = other.world_to_camera(self.C)  # Translation in cam2 frame

        # E = [t]_x @ R where [t]_x is skew-symmetric matrix
        t_skew = np.array(
            [
                [0, -t_rel[2], t_rel[1]],
                [t_rel[2], 0, -t_rel[0]],
                [-t_rel[1], t_rel[0], 0],
            ]
        )

        return t_skew @ R_rel

    def fundamental_matrix_to(
        self, other: "CameraPose", K1: np.ndarray, K2: np.ndarray
    ) -> np.ndarray:
        """
        Compute fundamental matrix F from this camera to another.

        Satisfies: p2^T @ F @ p1 = 0 for corresponding pixel coordinates

        Args:
            other: The second camera pose
            K1: Intrinsic matrix of this camera
            K2: Intrinsic matrix of other camera
        Returns:
            3x3 fundamental matrix
        """
        E = self.essential_matrix_to(other)
        return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

    # =========================================================================
    # COMPOSITION AND INVERSION
    # =========================================================================

    def compose(self, other: "CameraPose") -> "CameraPose":
        """
        Compose this pose with another: self @ other.

        If self transforms A->B and other transforms B->C,
        result transforms A->C.
        """
        R_new = self.R @ other.R
        C_new = self.camera_to_world(other.C)
        return CameraPose(R_new, C_new)

    def inverse(self) -> "CameraPose":
        """
        Get inverse pose.

        If this transforms world->camera,
        inverse transforms camera->world.
        """
        return CameraPose(self.R.T, self.t)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def copy(self) -> "CameraPose":
        """Create a copy of this pose."""
        return CameraPose(self.R.copy(), self.C.copy())

    def __repr__(self) -> str:
        return (
            f"CameraPose(\n"
            f"  C={self.C},\n"
            f"  euler(deg)={self.euler_angles_deg},\n"
            f"  rvec={self.rvec},\n"
            f"  tvec={self.tvec}\n)"
        )

    def print_summary(self):
        """Print concise pose summary."""
        print(f"Camera Center: {np.round(self.C, 3)}")
        print(f"Euler (deg):   {np.round(self.euler_angles_deg, 2)}")
        print(f"Optical Axis:  {np.round(self.get_optical_axis(), 3)}")

    def print_full_info(self):
        """Print comprehensive pose information for debugging."""
        print("=" * 60)
        print("CAMERA POSE INFORMATION")
        print("=" * 60)
        print(f"\nCamera Center (world coords):\n  C = {self.C}")
        print(f"\nRotation Matrix R (world to camera):\n{self.R}")
        print(f"\nTranslation Vector:\n  t = {self.t}")
        print(f"\nRodrigues Vector:\n  rvec = {self.rvec}")
        print(f"\nEuler Angles (XYZ, degrees):\n  {self.euler_angles_deg}")
        print(f"\nRotation: {self.rotation_angle_deg:.2f}° about {self.rotation_axis}")
        print(f"\nOptical Axis (world):\n  {self.get_optical_axis()}")
        print(f"\nExtrinsic Matrix [R|t]:\n{self.extrinsic_matrix_3x4}")
        print("=" * 60)


# =============================================================================
# QUICK EXAM PROBLEM SOLVER
# =============================================================================


def solve_exam_problem(rvec, tvec, P_cam):
    """
    Quick solver for the common exam pattern:
    Given PnP output and camera-frame point, find world coordinates.

    Args:
        rvec: Rodrigues vector from PnP
        tvec: Translation vector from PnP
        P_cam: Point in camera coordinates

    Returns:
        P_world: Point in world coordinates
    """
    pose = CameraPose.from_rvec_tvec(rvec, tvec)
    P_world = pose.camera_to_world(P_cam)
    return P_world


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE: Solving Exam Question")
    print("=" * 60)

    # Given from PnP solver
    rvec = np.array([-0.05, -1.51, -0.00])
    tvec = np.array([87.39, -2.25, -24.89])

    # Point from stereo reconstruction (camera coords)
    P_cam = np.array([-6.71, 0.23, 21.59])

    # Method 1: Quick solver
    P_world = solve_exam_problem(rvec, tvec, P_cam)
    print(f"\nQuick solve result: {np.round(P_world, 2)}")

    # Method 2: Using the class
    pose = CameraPose.from_rvec_tvec(rvec, tvec)
    pose.print_summary()

    print(f"\nP_cam = {P_cam}")
    print(f"P_world = {pose.camera_to_world(P_cam)}")
    print(f"P_world (rounded) = {np.round(pose.camera_to_world(P_cam), 2)}")

    # Verify round-trip
    P_cam_verify = pose.world_to_camera(P_world)
    print(f"\nVerification (back to camera): {np.round(P_cam_verify, 2)}")

    print("\n" + "=" * 60)
    print("ANSWER: 40.71, -1.98, 96.75")
    print("=" * 60)
