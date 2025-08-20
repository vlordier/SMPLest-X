#!/usr/bin/env python3
"""
Compare multiple meshes to validate all are unique
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

def compare_mesh_pair(vertices1, vertices2, name1, name2):
    """Compare two sets of vertices"""
    vertex_diff = vertices1 - vertices2
    max_diff = np.max(np.abs(vertex_diff))
    mean_diff = np.mean(np.abs(vertex_diff))
    
    centroid1 = np.mean(vertices1, axis=0)
    centroid2 = np.mean(vertices2, axis=0)
    centroid_distance = np.linalg.norm(centroid1 - centroid2)
    
    return {
        'names': (name1, name2),
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'centroid_distance': centroid_distance,
        'is_different': max_diff > 1e-6
    }

def main():
    mesh_dir = Path("/Users/vincent.lordier/Work/SMPLest-X/demo/output_meshes/1349093_720p")
    ply_files = list(mesh_dir.glob("*.ply"))[:5]  # Get first 5 meshes
    ply_files.sort()
    
    print("🔬 MULTIPLE MESH COMPARISON")
    print("=" * 50)
    
    # Load all meshes
    meshes = {}
    for ply_file in ply_files:
        vertices = read_ply_vertices(ply_file)
        meshes[ply_file.stem] = vertices
        print(f"📁 Loaded {ply_file.stem}: {len(vertices)} vertices")
    
    print("\n📊 PAIRWISE COMPARISONS:")
    print("-" * 80)
    print(f"{'Mesh 1':<20} {'Mesh 2':<20} {'Max Diff':<12} {'Mean Diff':<12} {'Centroid Dist':<15} {'Unique'}")
    print("-" * 80)
    
    comparisons = []
    mesh_names = list(meshes.keys())
    
    for i in range(len(mesh_names)):
        for j in range(i+1, len(mesh_names)):
            name1, name2 = mesh_names[i], mesh_names[j]
            vertices1, vertices2 = meshes[name1], meshes[name2]
            
            result = compare_mesh_pair(vertices1, vertices2, name1, name2)
            comparisons.append(result)
            
            status = "✅ YES" if result['is_different'] else "❌ NO"
            print(f"{name1:<20} {name2:<20} {result['max_diff']:<12.6f} {result['mean_diff']:<12.6f} {result['centroid_distance']:<15.6f} {status}")
    
    print("-" * 80)
    
    # Summary
    unique_pairs = sum(1 for c in comparisons if c['is_different'])
    total_pairs = len(comparisons)
    
    print("\n📈 SUMMARY:")
    print(f"   Total mesh pairs compared: {total_pairs}")
    print(f"   Unique pairs found: {unique_pairs}")
    print(f"   Identical pairs: {total_pairs - unique_pairs}")
    print(f"   Uniqueness rate: {unique_pairs/total_pairs*100:.1f}%")
    
    if unique_pairs == total_pairs:
        print("\n🎉 SUCCESS: All meshes are unique!")
        print("✅ Inference pipeline generates distinct meshes for each frame")
    else:
        print(f"\n⚠️  WARNING: {total_pairs - unique_pairs} identical mesh pairs found")
        print("❌ Some frames may be producing identical results")
    
    # Show range of differences
    max_diffs = [c['max_diff'] for c in comparisons]
    print("\n📏 DIFFERENCE STATISTICS:")
    print(f"   Minimum difference: {min(max_diffs):.6f}")
    print(f"   Maximum difference: {max(max_diffs):.6f}")
    print(f"   Average difference: {np.mean(max_diffs):.6f}")
    
if __name__ == "__main__":
    main()