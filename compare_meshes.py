#!/usr/bin/env python3
"""
Script to compare two PLY mesh files and validate they are different.
This ensures our inference pipeline generates unique meshes per frame.
"""

import numpy as np
from pathlib import Path

def read_ply_vertices(ply_path):
    """Read vertices from PLY file"""
    vertices = []
    
    with open(ply_path, 'r') as f:
        lines = f.readlines()
        
        # Find number of vertices from header
        vertex_count = 0
        header_end = 0
        
        for i, line in enumerate(lines):
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('end_header'):
                header_end = i + 1
                break
        
        # Read vertex data
        for i in range(header_end, header_end + vertex_count):
            coords = lines[i].strip().split()
            vertices.append([float(coords[0]), float(coords[1]), float(coords[2])])
    
    return np.array(vertices)

def compare_meshes(mesh1_path, mesh2_path):
    """Compare two meshes and return detailed statistics"""
    print("🔍 Comparing meshes:")
    print(f"   Mesh 1: {mesh1_path}")
    print(f"   Mesh 2: {mesh2_path}")
    print()
    
    # Read vertices
    vertices1 = read_ply_vertices(mesh1_path)
    vertices2 = read_ply_vertices(mesh2_path)
    
    print("📊 Mesh Statistics:")
    print(f"   Mesh 1 vertices: {len(vertices1)}")
    print(f"   Mesh 2 vertices: {len(vertices2)}")
    print()
    
    if len(vertices1) != len(vertices2):
        print("❌ Meshes have different vertex counts!")
        return False
    
    # Calculate differences
    vertex_diff = vertices1 - vertices2
    max_diff = np.max(np.abs(vertex_diff))
    mean_diff = np.mean(np.abs(vertex_diff))
    
    # Calculate distance between mesh centroids
    centroid1 = np.mean(vertices1, axis=0)
    centroid2 = np.mean(vertices2, axis=0)
    centroid_distance = np.linalg.norm(centroid1 - centroid2)
    
    # Calculate bounding box differences
    bbox1_min, bbox1_max = np.min(vertices1, axis=0), np.max(vertices1, axis=0)
    bbox2_min, bbox2_max = np.min(vertices2, axis=0), np.max(vertices2, axis=0)
    bbox_diff = np.max(np.abs(bbox1_max - bbox2_max)) + np.max(np.abs(bbox1_min - bbox2_min))
    
    print("📏 Difference Analysis:")
    print(f"   Maximum vertex difference: {max_diff:.6f}")
    print(f"   Mean vertex difference: {mean_diff:.6f}")
    print(f"   Centroid distance: {centroid_distance:.6f}")
    print(f"   Bounding box difference: {bbox_diff:.6f}")
    print()
    
    print("🎯 Mesh Centroids:")
    print(f"   Mesh 1: [{centroid1[0]:.3f}, {centroid1[1]:.3f}, {centroid1[2]:.3f}]")
    print(f"   Mesh 2: [{centroid2[0]:.3f}, {centroid2[1]:.3f}, {centroid2[2]:.3f}]")
    print()
    
    print("📦 Bounding Boxes:")
    print(f"   Mesh 1: [{bbox1_min[0]:.3f}, {bbox1_min[1]:.3f}, {bbox1_min[2]:.3f}] to [{bbox1_max[0]:.3f}, {bbox1_max[1]:.3f}, {bbox1_max[2]:.3f}]")
    print(f"   Mesh 2: [{bbox2_min[0]:.3f}, {bbox2_min[1]:.3f}, {bbox2_min[2]:.3f}] to [{bbox2_max[0]:.3f}, {bbox2_max[1]:.3f}, {bbox2_max[2]:.3f}]")
    print()
    
    # Check if meshes are meaningfully different
    threshold = 1e-6  # Very small threshold for floating point comparison
    
    if max_diff < threshold:
        print("❌ IDENTICAL MESHES - No significant differences found!")
        print(f"   Maximum difference {max_diff:.9f} is below threshold {threshold}")
        return False
    else:
        print("✅ UNIQUE MESHES - Significant differences found!")
        print(f"   Maximum difference {max_diff:.6f} is above threshold {threshold}")
        
        # Additional checks for meaningful differences
        if mean_diff > 0.001:  # 1mm difference
            print("✅ Large mean difference - meshes represent different poses")
        elif centroid_distance > 0.01:  # 1cm centroid movement
            print("✅ Significant centroid movement - different body positions")
        elif bbox_diff > 0.01:  # 1cm bounding box difference
            print("✅ Different bounding boxes - different body poses/sizes")
        else:
            print("⚠️  Small differences - might be minor pose variations")
        
        return True

def main():
    # Use existing mesh files
    mesh_dir = Path("/Users/vincent.lordier/Work/SMPLest-X/demo/output_meshes/1349093_720p")
    
    # Find first two PLY files
    ply_files = list(mesh_dir.glob("*.ply"))
    
    if len(ply_files) < 2:
        print("❌ Need at least 2 PLY files to compare")
        return
    
    # Sort files and compare first two
    ply_files.sort()
    mesh1 = ply_files[0] 
    mesh2 = ply_files[1]
    
    print("=" * 60)
    print("🔬 MESH COMPARISON ANALYSIS")
    print("=" * 60)
    print()
    
    are_different = compare_meshes(mesh1, mesh2)
    
    print("=" * 60)
    if are_different:
        print("🎉 SUCCESS: Meshes are unique and different!")
        print("✅ Inference pipeline is generating distinct meshes per frame")
    else:
        print("❌ ISSUE: Meshes appear to be identical!")
        print("⚠️  This suggests inference might not be working properly")
    print("=" * 60)

if __name__ == "__main__":
    main()