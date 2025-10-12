"""
File format conversion utilities (PLY, Splat, etc.)
"""
from pathlib import Path


def points3d_to_ply(points3d_txt: Path, ply_path: Path) -> None:
    """
    Convert COLMAP points3D.txt to PLY format

    Args:
        points3d_txt: Path to COLMAP points3D.txt file
        ply_path: Output PLY file path
    """
    with open(points3d_txt, 'r') as f_txt, open(ply_path, 'w') as f_ply:
        lines = f_txt.readlines()

        # Parse valid data lines (skip comments)
        points = []
        for line in lines:
            if line.strip().startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            # points3D.txt format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]...
            if len(parts) >= 7:
                X, Y, Z = map(float, parts[1:4])
                R, G, B = map(int, parts[4:7])
                points.append((X, Y, Z, R, G, B))

        # Write PLY header
        f_ply.write("ply\nformat ascii 1.0\n")
        f_ply.write(f"element vertex {len(points)}\n")
        f_ply.write("property float x\nproperty float y\nproperty float z\n")
        f_ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f_ply.write("end_header\n")

        # Write point data
        for (x, y, z, r, g, b) in points:
            f_ply.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
