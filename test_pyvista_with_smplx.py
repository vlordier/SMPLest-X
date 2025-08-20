#!/usr/bin/env python3
"""
Test PyVista renderer with actual SMPL-X mesh.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from utils.pyvista_renderer import render_smplx_mesh

def read_ply_mesh(filepath):
    """Read PLY file and return vertices and faces."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        vertex_count = 0
        face_count = 0
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif line == 'end_header':
                data_start = i + 1
                break
        
        # Read vertices
        vertices = []
        for i in range(data_start, data_start + vertex_count):
            parts = lines[i].strip().split()[:3]
            vertices.append([float(p) for p in parts])
        
        # Read faces  
        faces = []
        for i in range(data_start + vertex_count, data_start + vertex_count + face_count):
            parts = lines[i].strip().split()
            if int(parts[0]) == 3:  # Triangular face
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
        
        return np.array(vertices), np.array(faces)
        
    except Exception as e:
        print(f"❌ Error reading PLY: {e}")
        return None, None

def main():
    print("🎨 Testing PyVista with SMPL-X mesh...")
    
    # Try to load a generated mesh
    mesh_path = 'demo/output_meshes/1349093_720p/000001_person0.ply'
    
    vertices, faces = read_ply_mesh(mesh_path)
    
    if vertices is None:
        print(f"❌ Could not load mesh from {mesh_path}")
        return
    
    print(f"✅ Loaded mesh: {len(vertices):,} vertices, {len(faces):,} faces")
    
    # Test different rendering styles
    test_cases = [
        ('shaded', 'pyvista_smplx_shaded.png', 'lightblue'),
        ('wireframe', 'pyvista_smplx_wireframe.png', 'black'),
        ('both', 'pyvista_smplx_both.png', 'lightcoral'),
    ]
    
    for renderer_type, output_file, color in test_cases:
        print(f"🎨 Rendering {renderer_type}...")
        
        success = render_smplx_mesh(
            vertices, faces,
            output_path=output_file,
            renderer_type=renderer_type,
            camera_position='iso',
            color=color,
            window_size=(1024, 1024)
        )
        
        if success:
            print(f"✅ {renderer_type.capitalize()} render saved: {output_file}")
        else:
            print(f"❌ {renderer_type.capitalize()} render failed")
    
    print("🎉 PyVista SMPL-X rendering test complete!")

if __name__ == "__main__":
    main()