import open3d as o3d
import numpy as np
import copy


# helper function for drawing
# If you want it to be more clear set recolor=True
def draw_registrations(source, target, transformation=None, recolor=False):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    if recolor:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    if transformation is not None:
        source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


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
draw_registrations(source, target, result.transformation, True)


## RANSAC
