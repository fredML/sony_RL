"""
romiscan.proc3d
---------------
This module contains all functions for processing of 3D data.
"""

import open3d
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

import cv2


Vector3dVector = open3d.utility.Vector3dVector
Vector3iVector = open3d.utility.Vector3iVector
Vector2iVector = open3d.utility.Vector2iVector
PointCloud = open3d.geometry.PointCloud
TriangleMesh = open3d.geometry.TriangleMesh


def index2point(indexes, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points
    Parameters
    ----------
    indexes : np.ndarray
        Nxd array of indices
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels
    Returns
    -------
    np.ndarray
        Nxd array of points
    """
    return voxel_size * indexes + origin[np.newaxis, :]


def point2index(points, origin, voxel_size):
    """Converts discrete nd indexes to a 3d points
    Parameters
    ----------
    points : np.ndarray
        Nxd array of points
    origin : np.ndarray
        1d array of length d
    voxel_size : float
        size of voxels
    Returns
    -------
    np.ndarray (dtype=int)
        Nxd array of indices
    """
    return np.array(np.round((points - origin[np.newaxis, :]) / voxel_size), dtype=int)


def vol2pcd(volume, origin, voxel_size, level_set_value=0):
    """Converts a binary volume into a point cloud with normals.
    Parameters
    ----------
    volume : np.ndarray
        NxMxP 3D binary numpy array
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled
    Returns
    -------
    open3d.geometry.PointCloud
    """
    volume = 1.0*(volume > 0.5)  # variable level ?
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1-volume)
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    gx, gy, gz = np.gradient(dist)
    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    on_edge = (dist > -level_set_value) * (dist <= -level_set_value+np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    pts = np.zeros((0, 3))
    normals = np.zeros((0, 3))
    for i in tqdm(range(len(x)), desc="Computing normals"):
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3)/2
            pts = np.vstack([pts, np.array([x[i] - grad_normalized[0] * val,
                                            y[i] - grad_normalized[1] * val,
                                            z[i] - grad_normalized[2] * val])])
            normals = np.vstack([normals, -np.array([grad_normalized[0],
                                                     grad_normalized[1],
                                                     grad_normalized[2]])])

    pts = index2point(pts, origin, voxel_size)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    pcd.normals = Vector3dVector(normals)
    pcd.normalize_normals()

    return pcd


def vol2pcd_exp(volume, origin, voxel_size, level_set_value=0):
    """Converts a binary volume into a point cloud with normals.
    Parameters
    ----------
    volume : np.ndarray
        NxMxP 3D binary numpy array
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled
    Returns
    -------
    open3d.geometry.PointCloud
    """
    # volume = 1.0*(volume>0.5) # variable level ?

    binary_volume = (volume > 0).astype('uint8')

    dist = np.stack([cv2.distanceTransform(binary_volume[:, :, k], distanceType=2,
                    maskSize=5) for k in range(binary_volume.shape[-1])], axis=-1)

    mdist = np.stack([cv2.distanceTransform(1-binary_volume[:, :, k], distanceType=2,
                     maskSize=5) for k in range(binary_volume.shape[-1])], axis=-1)

    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    on_edge = (dist > -level_set_value) * (dist <= -level_set_value+np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    pts = np.array([x, y, z]).T

    pts = index2point(pts, origin, voxel_size)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    return pcd


'''def vol2pcd_exp(volume, origin, voxel_size, level_set_value=0):
    """Converts a binary volume into a point cloud with normals.
    Parameters
    ----------
    volume : np.ndarray
        NxMxP 3D binary numpy array
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled
    Returns
    -------
    open3d.geometry.PointCloud
    """
    volume = 1.0*(volume>0.5) # variable level ?
    dist = distance_transform_edt(volume)
    mdist = distance_transform_edt(1-volume)
    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    on_edge = (dist > -level_set_value) * (dist <= -level_set_value+np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    pts=np.array([x,y,z]).T
    
    pts = index2point(pts, origin, voxel_size)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    return pcd'''

def vol2pcd2(volume, origin, voxel_size, level_set_value=0):
    """Converts a volume into a point cloud with normals.
    Parameters
    ----------
    volume : numpy.ndarray
        NxMxP 3D numpy array
    origin : numpy.ndarray
        origin of the volume
    voxel_size: float
        voxel size
    level_set_value: float
        distance of the level set on which the points are sampled
    Returns
    -------
    open3d.geometry.PointCloud
        Point-cloud with normal vectors.
    """
    # volume = 1.0 * (volume > 0.5)  # variable level ?
    volume = (volume > 0.5).astype(np.uint8)
    dist = np.stack([cv2.distanceTransform(volume[:, :, k], distanceType=2,
                    maskSize=5) for k in range(volume.shape[-1])], axis=-1)
    mdist = np.stack([cv2.distanceTransform(1-volume[:, :, k], distanceType=2,
                     maskSize=5) for k in range(volume.shape[-1])], axis=-1)

    dist = np.where(dist > 0.5, dist - 0.5, -mdist + 0.5)

    gx, gy, gz = np.gradient(dist)

    gx = gaussian_filter(gx, 1)
    gy = gaussian_filter(gy, 1)
    gz = gaussian_filter(gz, 1)

    on_edge = (dist > -level_set_value) * \
        (dist <= -level_set_value + np.sqrt(3))
    x, y, z = np.nonzero(on_edge)

    def _compute_normal(i):
        p_i, normal_i = np.array([np.nan, np.nan, np.nan]), np.array(
            [np.nan, np.nan, np.nan])
        grad = np.array([gx[x[i], y[i], z[i]],
                         gy[x[i], y[i], z[i]],
                         gz[x[i], y[i], z[i]]])
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            grad_normalized = grad / grad_norm
            val = dist[x[i], y[i], z[i]] + level_set_value - np.sqrt(3) / 2
            p_i = np.array([x[i] - grad_normalized[0] * val,
                            y[i] - grad_normalized[1] * val,
                            z[i] - grad_normalized[2] * val])
            normal_i = -np.array([grad_normalized[0],
                                  grad_normalized[1],
                                  grad_normalized[2]])
        return p_i, normal_i

    from joblib import Parallel
    from joblib import delayed
    all_norms = Parallel(n_jobs=-1)(
        delayed(_compute_normal)(i) for i in tqdm(range(len(x)), desc="Computing point normals"))

    pts, normals = zip(*all_norms)
    not_none_idx = np.where(~np.isnan(normals).any(axis=1))[
        0]  # Detect np.nan (if grad_norm > 0)
    # Keep points with a positive gradiant norm
    pts = np.array(pts)[not_none_idx]
    # Keep normals with a positive gradiant norm
    normals = np.array(normals)[not_none_idx]

    pts = index2point(pts, origin, voxel_size)
    pcd = PointCloud()
    pcd.points = Vector3dVector(pts)
    pcd.normals = Vector3dVector(normals)
    pcd.normalize_normals()

    return pcd
