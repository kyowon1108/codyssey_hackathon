"""
File format conversion utilities (PLY, Splat, etc.)
"""
import numpy as np
import struct
from pathlib import Path
from plyfile import PlyData
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


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


def convert_ply_to_splat(ply_path: Path, splat_path: Path) -> None:
    """
    Convert PLY file to splat format for antimatter15/splat viewer

    .splat format stores each gaussian as:
    - position (3 floats: x, y, z)
    - scale (3 floats: sx, sy, sz)
    - color (4 bytes: r, g, b, opacity)
    - rotation (4 floats: quaternion)

    Args:
        ply_path: Input PLY file path
        splat_path: Output splat file path
    """
    logger.info(f"Loading PLY from {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    num_points = len(vertex)
    logger.info(f"Converting {num_points} gaussians to splat format...")

    # splat file generation
    with open(splat_path, 'wb') as f:
        for i in range(num_points):
            if i % 10000 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{num_points} ({100*i/num_points:.1f}%)")

            # Position
            x = float(vertex['x'][i])
            y = float(vertex['y'][i])
            z = float(vertex['z'][i])

            # Scale (log scale in PLY)
            if 'scale_0' in vertex:
                sx = np.exp(float(vertex['scale_0'][i]))
                sy = np.exp(float(vertex['scale_1'][i]))
                sz = np.exp(float(vertex['scale_2'][i]))
            else:
                # Default values
                sx = sy = sz = 0.01

            # Color (SH coefficients -> RGB)
            if 'f_dc_0' in vertex:
                # SH DC component to RGB
                r = float(vertex['f_dc_0'][i]) * 0.28209479177387814 + 0.5
                g = float(vertex['f_dc_1'][i]) * 0.28209479177387814 + 0.5
                b = float(vertex['f_dc_2'][i]) * 0.28209479177387814 + 0.5
            elif 'red' in vertex:
                r = float(vertex['red'][i]) / 255.0
                g = float(vertex['green'][i]) / 255.0
                b = float(vertex['blue'][i]) / 255.0
            else:
                r = g = b = 0.5

            # Clamp to [0, 1]
            r = max(0, min(1, r))
            g = max(0, min(1, g))
            b = max(0, min(1, b))

            # Opacity
            if 'opacity' in vertex:
                opacity = 1.0 / (1.0 + np.exp(-float(vertex['opacity'][i])))
            else:
                opacity = 1.0

            # Rotation (quaternion)
            if 'rot_0' in vertex:
                qw = float(vertex['rot_0'][i])
                qx = float(vertex['rot_1'][i])
                qy = float(vertex['rot_2'][i])
                qz = float(vertex['rot_3'][i])
                # Normalize
                norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                if norm > 0:
                    qw /= norm
                    qx /= norm
                    qy /= norm
                    qz /= norm
            else:
                # Identity quaternion
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

            # Write to file
            # Format: position(3f) + scale(3f) + color(4B) + rotation(4f)
            f.write(struct.pack('fff', x, y, z))  # position
            f.write(struct.pack('fff', sx, sy, sz))  # scale
            f.write(struct.pack('BBBB',
                               int(r * 255),
                               int(g * 255),
                               int(b * 255),
                               int(opacity * 255)))  # color + opacity
            f.write(struct.pack('ffff', qw, qx, qy, qz))  # rotation

    file_size_mb = Path(splat_path).stat().st_size / (1024*1024)
    logger.info(f"Conversion complete! Output: {splat_path} ({file_size_mb:.1f} MB)")
