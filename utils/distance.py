"""
This file contains implementations for calculating distance between point clouds.
As GPU is not available it contains simple CPU implementations, once the GPU is available,
CUDA accelerated methods will be used.
"""
import numpy as np
from sklearn.neighbors import KDTree


def chamfer_distance_sklearn(array1, array2):
    batch_size, num_point = array1.shape[:2]
    dist = 0
    for i in range(batch_size):
        tree1 = KDTree(array1[i], leaf_size=num_point+1)
        tree2 = KDTree(array2[i], leaf_size=num_point+1)
        distances1, _ = tree1.query(array2[i])
        distances2, _ = tree2.query(array1[i])
        av_dist1 = np.mean(distances1)
        av_dist2 = np.mean(distances2)
        dist = dist + (av_dist1+av_dist2)/batch_size
    return dist
