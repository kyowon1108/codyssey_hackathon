#!/usr/bin/env python3
"""
Filter outlier Gaussians from PLY file based on distance from neighbors.
"""
import numpy as np
import sys
from pathlib import Path
from plyfile import PlyData, PlyElement
from sklearn.neighbors import NearestNeighbors

def filter_outliers(ply_path, output_path, k_neighbors=20, std_threshold=2.0):
    """
    Filter outlier Gaussians based on distance to k nearest neighbors.
    
    Args:
        ply_path: Input PLY file path
        output_path: Output filtered PLY file path
        k_neighbors: Number of neighbors to consider (default: 20)
        std_threshold: Standard deviation threshold for outlier detection (default: 2.0)
    """
    print(f"Loading PLY from {ply_path}...")
    plydata = PlyData.read(ply_path)
    
    vertex = plydata['vertex']
    total_gaussians = len(vertex)
    print(f"Total Gaussians: {total_gaussians}")
    
    # Extract positions
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Compute k-nearest neighbors
    print(f"Computing {k_neighbors} nearest neighbors...")
    nbrs = NearestNeighbors(n_neighbors=k_neighbors+1, algorithm='auto').fit(positions)
    distances, indices = nbrs.kneighbors(positions)
    
    # Use mean distance to k-nearest neighbors (excluding self)
    mean_distances = distances[:, 1:].mean(axis=1)
    
    # Filter based on statistical threshold
    mean_dist = mean_distances.mean()
    std_dist = mean_distances.std()
    threshold = mean_dist + std_threshold * std_dist
    
    print(f"Mean distance: {mean_dist:.4f}")
    print(f"Std distance: {std_dist:.4f}")
    print(f"Threshold: {threshold:.4f}")
    
    # Create mask for inliers
    inlier_mask = mean_distances < threshold
    num_inliers = inlier_mask.sum()
    num_outliers = total_gaussians - num_inliers
    
    print(f"Inliers: {num_inliers} ({100*num_inliers/total_gaussians:.1f}%)")
    print(f"Outliers removed: {num_outliers} ({100*num_outliers/total_gaussians:.1f}%)")
    
    # Filter vertex data
    filtered_vertex = vertex[inlier_mask]
    
    # Create new PLY element
    filtered_element = PlyElement.describe(filtered_vertex, 'vertex')
    
    # Write filtered PLY
    print(f"Writing filtered PLY to {output_path}...")
    PlyData([filtered_element]).write(output_path)
    print("Done!")
    
    return num_inliers, num_outliers

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python filter_outliers.py <input.ply> <output.ply> [k_neighbors] [std_threshold]")
        print("  k_neighbors: Number of neighbors to consider (default: 20)")
        print("  std_threshold: Std dev threshold for outliers (default: 2.0)")
        sys.exit(1)
    
    input_ply = Path(sys.argv[1])
    output_ply = Path(sys.argv[2])
    k_neighbors = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    std_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    
    if not input_ply.exists():
        print(f"Error: Input file not found: {input_ply}")
        sys.exit(1)
    
    filter_outliers(input_ply, output_ply, k_neighbors, std_threshold)
