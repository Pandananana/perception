import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class CameraMatrix:
    def __init__(self, numpy_array: np.ndarray) -> None:
        if numpy_array.shape != (3, 3):
            raise ValueError("Must be a 3x3 array")

        # Store values as given - don't modify them
        # Uses positive focal lengths
        self.fx = float(numpy_array[0, 0])
        self.fy = float(numpy_array[1, 1])
        self.cx = float(numpy_array[0, 2])  # More standard naming
        self.cy = float(numpy_array[1, 2])

        # Keep the full matrix for matrix operations
        self.K = numpy_array.astype(float)

    def project(self, P_cam: np.ndarray) -> tuple[float, float]:
        """Project a 3D camera-frame point to 2D pixel coordinates."""
        X, Y, Z = P_cam
        x = (self.fx * X / Z) + self.cx
        y = (self.fy * Y / Z) + self.cy
        return (x, y)


class CameraPose:
    def __init__(self, rotation: np.ndarray, position: np.ndarray):
        self.R = rotation  # 3x3 rotation matrix
        self.C = position  # Camera center in world coords

    def world_to_camera(self, P_world: np.ndarray) -> np.ndarray:
        """Transform point from world frame to camera frame."""
        return self.R @ (P_world - self.C)


class SimilarityMetrics:
    """Similarity metrics for image patch comparison."""

    @staticmethod
    def sad(A: np.ndarray, B: np.ndarray) -> float:
        """
        Sum of Absolute Differences.

        Args:
            A: First image patch
            B: Second image patch

        Returns:
            SAD score (lower is more similar)
        """
        return float(np.sum(np.abs(A - B)))

    @staticmethod
    def ssd(A: np.ndarray, B: np.ndarray) -> float:
        """
        Sum of Squared Differences.

        Args:
            A: First image patch
            B: Second image patch

        Returns:
            SSD score (lower is more similar)
        """
        return float(np.sum((A - B) ** 2))

    @staticmethod
    def ncc(A: np.ndarray, B: np.ndarray) -> float:
        """
        Normalized Cross-Correlation.

        Args:
            A: First image patch
            B: Second image patch

        Returns:
            NCC score in range [-1, 1] (higher is more similar, 1 is perfect match)
        """
        # Normalize by subtracting mean
        A_normalized = A - np.mean(A)
        B_normalized = B - np.mean(B)

        # Compute correlation
        numerator = np.sum(A_normalized * B_normalized)
        denominator = np.sqrt(np.sum(A_normalized**2) * np.sum(B_normalized**2))

        # Avoid division by zero
        if denominator == 0:
            return 0.0

        return float(numerator / denominator)


def plot_elbow_graph(data: np.ndarray, min_clusters: int = 3, max_clusters: int = 6) -> None:
    """
    Plot an elbow graph for K-means clustering.

    Args:
        data: Input data array where each row is a data point
        min_clusters: Minimum number of clusters to test
        max_clusters: Maximum number of clusters to test
    """
    k_values = range(min_clusters, max_clusters + 1)
    inertias = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure()
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.title('Elbow Method')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()
