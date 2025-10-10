#!/usr/bin/env python3
"""
Gaussian Splatting PLY를 .splat 포맷으로 변환
antimatter15/splat 뷰어 형식 사용
"""
import numpy as np
import struct
from pathlib import Path
from plyfile import PlyData

def convert_ply_to_splat(ply_path, splat_path):
    """
    PLY 파일을 splat 형식으로 변환
    .splat 형식은 각 gaussian을 다음과 같이 저장:
    - position (3 floats: x, y, z)
    - scale (3 floats: sx, sy, sz)
    - color (4 bytes: r, g, b, opacity)
    - rotation (4 floats: quaternion)
    """
    print(f"Loading PLY from {ply_path}...")
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    num_points = len(vertex)
    print(f"Processing {num_points} gaussians...")

    # splat 파일 생성
    with open(splat_path, 'wb') as f:
        for i in range(num_points):
            if i % 10000 == 0:
                print(f"  {i}/{num_points} ({100*i/num_points:.1f}%)")

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
                # 기본값
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

    print(f"Conversion complete! Output: {splat_path}")
    print(f"File size: {Path(splat_path).stat().st_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python convert_to_splat.py <ply_file> <splat_file>")
        sys.exit(1)

    ply_path = Path(sys.argv[1])
    splat_path = Path(sys.argv[2])

    if not ply_path.exists():
        print(f"Error: PLY file not found: {ply_path}")
        sys.exit(1)

    convert_ply_to_splat(ply_path, splat_path)
