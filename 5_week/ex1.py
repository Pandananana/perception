import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


# helper function for drawing using matplotlib (Open3D viz crashes on macOS)
def draw_registrations(source, target, transformation=None, recolor=False):
    # Get points as numpy arrays (avoid deepcopy which can crash)
    src_pts = np.asarray(source.points).copy()
    tgt_pts = np.asarray(target.points).copy()

    if transformation is not None:
        # Apply transformation manually: p' = R @ p + t
        src_pts = (transformation[:3, :3] @ src_pts.T).T + transformation[:3, 3]

    # Use matplotlib for 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for performance (plot every nth point)
    step = max(1, len(src_pts) // 5000)
    ax.scatter(
        src_pts[::step, 0],
        src_pts[::step, 1],
        src_pts[::step, 2],
        c="orange",
        s=1,
        label="Source (transformed)",
    )
    step = max(1, len(tgt_pts) // 5000)
    ax.scatter(
        tgt_pts[::step, 0],
        tgt_pts[::step, 1],
        tgt_pts[::step, 2],
        c="deepskyblue",
        s=1,
        label="Target",
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Point Cloud Registration Result")
    plt.show()


#
# Pre Exercises
#

source = o3d.io.read_point_cloud("ICP/r1.pcd")
target = o3d.io.read_point_cloud("ICP/r2.pcd")

# Used for downsampling.
voxel_size = 0.05

# Show models side by side
# draw_registrations(source, target)


## Finding Features in Pointclouds
source = source.voxel_down_sample(voxel_size)
target = target.voxel_down_sample(voxel_size)

# Estimate normals
source.estimate_normals()
target.estimate_normals()
# o3d.visualization.draw_geometries([source, target], point_show_normal=True)

# Compute FPFH descriptors
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)

point_to_point = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
point_to_plane = o3d.pipelines.registration.TransformationEstimationPointToPlane()
distance_threshold = 0.075

# Perform RANSAC registration
result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source, target, source_fpfh, target_fpfh, True, distance_threshold, point_to_point
)

# Visualize registration
# draw_registrations(source, target, result.transformation, True)


#
# Exercise A
#

r3 = o3d.io.read_point_cloud("ICP/r3.pcd")
r3 = r3.voxel_down_sample(voxel_size)
r3.estimate_normals()
r3_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    r3, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100)
)

r3_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source, r3, source_fpfh, r3_fpfh, True, distance_threshold, point_to_point
)

draw_registrations(source, r3, r3_result.transformation, True)
