#!/usr/bin/env python3
"""
Check mesh topology and verify SMPL-X compatibility.
Analyzes vertex counts, face counts, and mesh structure.
"""

import numpy as np
import json
from pathlib import Path
import sys

def read_ply_file(filepath):
    """Read PLY file and extract vertices and faces"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find header info
        vertex_count = 0
        face_count = 0
        in_header = True
        data_start = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
            elif line == 'end_header':
                in_header = False
                data_start = i + 1
                break
        
        # Read vertices
        vertices = []
        for i in range(data_start, data_start + vertex_count):
            parts = lines[i].strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        # Read faces
        faces = []
        for i in range(data_start + vertex_count, data_start + vertex_count + face_count):
            parts = lines[i].strip().split()
            if int(parts[0]) == 3:  # Triangle
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        return vertices, faces, vertex_count, face_count
        
    except Exception as e:
        print(f"❌ Error reading PLY file {filepath}: {e}")
        return None, None, 0, 0

def read_obj_file(filepath):
    """Read OBJ file and extract vertices and faces"""
    try:
        vertices = []
        faces = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    # Handle 1-indexed faces (subtract 1 for 0-indexed)
                    face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                    faces.append(face)
        
        vertices = np.array(vertices)
        faces = np.array(faces)
        
        return vertices, faces, len(vertices), len(faces)
        
    except Exception as e:
        print(f"❌ Error reading OBJ file {filepath}: {e}")
        return None, None, 0, 0

def analyze_mesh_structure(vertices, faces):
    """Analyze mesh geometric properties"""
    if vertices is None or faces is None:
        return {}
    
    analysis = {}
    
    # Basic counts
    analysis['vertex_count'] = len(vertices)
    analysis['face_count'] = len(faces)
    
    # Vertex statistics
    analysis['vertex_bounds'] = {
        'min': vertices.min(axis=0).tolist(),
        'max': vertices.max(axis=0).tolist(),
        'center': vertices.mean(axis=0).tolist()
    }
    
    # Face validation
    max_vertex_index = vertices.shape[0] - 1
    invalid_faces = []
    for i, face in enumerate(faces):
        if any(idx < 0 or idx > max_vertex_index for idx in face):
            invalid_faces.append(i)
    
    analysis['invalid_faces'] = len(invalid_faces)
    analysis['valid_topology'] = len(invalid_faces) == 0
    
    # Check for triangular faces
    triangle_count = sum(1 for face in faces if len(face) == 3)
    analysis['triangle_faces'] = triangle_count
    analysis['all_triangular'] = triangle_count == len(faces)
    
    # Mesh dimensions
    bounds = analysis['vertex_bounds']
    dimensions = np.array(bounds['max']) - np.array(bounds['min'])
    analysis['dimensions'] = dimensions.tolist()
    analysis['mesh_size'] = np.linalg.norm(dimensions)
    
    return analysis

def check_smplx_compliance(analysis):
    """Check if mesh meets SMPL-X standards"""
    compliance = {
        'is_smplx_compliant': False,
        'issues': [],
        'warnings': []
    }
    
    # SMPL-X standard topology
    SMPLX_VERTICES = 10475
    SMPLX_FACES = 20908
    
    vertex_count = analysis.get('vertex_count', 0)
    face_count = analysis.get('face_count', 0)
    
    # Check vertex count
    if vertex_count != SMPLX_VERTICES:
        compliance['issues'].append(
            f"Vertex count mismatch: got {vertex_count}, expected {SMPLX_VERTICES}"
        )
    
    # Check face count
    if face_count != SMPLX_FACES:
        compliance['issues'].append(
            f"Face count mismatch: got {face_count}, expected {SMPLX_FACES}"
        )
    
    # Check topology validity
    if not analysis.get('valid_topology', False):
        compliance['issues'].append(
            f"Invalid topology: {analysis.get('invalid_faces', 0)} faces have invalid vertex indices"
        )
    
    # Check triangular faces
    if not analysis.get('all_triangular', False):
        compliance['issues'].append(
            f"Non-triangular faces detected: only {analysis.get('triangle_faces', 0)} of {face_count} are triangular"
        )
    
    # Check reasonable mesh size
    mesh_size = analysis.get('mesh_size', 0)
    if mesh_size < 0.1:
        compliance['warnings'].append(f"Very small mesh size: {mesh_size:.3f}")
    elif mesh_size > 100:
        compliance['warnings'].append(f"Very large mesh size: {mesh_size:.3f}")
    
    # Overall compliance
    compliance['is_smplx_compliant'] = len(compliance['issues']) == 0
    
    return compliance

def check_mesh_consistency(mesh_files):
    """Check consistency across multiple mesh files"""
    if len(mesh_files) < 2:
        return {"consistent": True, "message": "Only one mesh file to check"}
    
    reference_analysis = None
    inconsistencies = []
    
    for mesh_file in mesh_files[:5]:  # Check first 5 files
        if mesh_file.suffix.lower() == '.ply':
            vertices, faces, v_count, f_count = read_ply_file(mesh_file)
        elif mesh_file.suffix.lower() == '.obj':
            vertices, faces, v_count, f_count = read_obj_file(mesh_file)
        else:
            continue
            
        analysis = analyze_mesh_structure(vertices, faces)
        
        if reference_analysis is None:
            reference_analysis = analysis
            continue
        
        # Check consistency with reference
        if analysis['vertex_count'] != reference_analysis['vertex_count']:
            inconsistencies.append(f"{mesh_file.name}: vertex count {analysis['vertex_count']} != {reference_analysis['vertex_count']}")
        
        if analysis['face_count'] != reference_analysis['face_count']:
            inconsistencies.append(f"{mesh_file.name}: face count {analysis['face_count']} != {reference_analysis['face_count']}")
    
    return {
        "consistent": len(inconsistencies) == 0,
        "inconsistencies": inconsistencies,
        "reference_topology": {
            "vertices": reference_analysis['vertex_count'] if reference_analysis else 0,
            "faces": reference_analysis['face_count'] if reference_analysis else 0
        }
    }

def main():
    print("🔍 SMPL-X MESH TOPOLOGY CHECKER")
    print("=" * 50)
    
    # Check mesh directories
    project_dir = Path("/Users/vincent.lordier/Work/SMPLest-X")
    mesh_dir = project_dir / "demo" / "output_meshes" / "1349093_720p"
    
    if not mesh_dir.exists():
        print(f"❌ Mesh directory not found: {mesh_dir}")
        return
    
    # Find mesh files
    ply_files = list(mesh_dir.glob("*.ply"))
    obj_files = list(mesh_dir.glob("*.obj"))
    
    print(f"📁 Found mesh files:")
    print(f"   PLY files: {len(ply_files)}")
    print(f"   OBJ files: {len(obj_files)}")
    
    if not ply_files and not obj_files:
        print("❌ No mesh files found!")
        return
    
    # Analyze first PLY file in detail
    if ply_files:
        print(f"\n🔍 Analyzing PLY file: {ply_files[0].name}")
        vertices, faces, v_count, f_count = read_ply_file(ply_files[0])
        analysis = analyze_mesh_structure(vertices, faces)
        compliance = check_smplx_compliance(analysis)
        
        print(f"📊 Mesh Statistics:")
        print(f"   Vertices: {analysis.get('vertex_count', 0):,}")
        print(f"   Faces: {analysis.get('face_count', 0):,}")
        print(f"   Triangular faces: {analysis.get('triangle_faces', 0):,}")
        print(f"   Valid topology: {analysis.get('valid_topology', False)}")
        print(f"   Mesh size: {analysis.get('mesh_size', 0):.3f}")
        
        bounds = analysis.get('vertex_bounds', {})
        if bounds:
            print(f"   Bounds: [{bounds['min'][0]:.3f}, {bounds['min'][1]:.3f}, {bounds['min'][2]:.3f}] to [{bounds['max'][0]:.3f}, {bounds['max'][1]:.3f}, {bounds['max'][2]:.3f}]")
        
        print(f"\n🎯 SMPL-X Compliance:")
        if compliance['is_smplx_compliant']:
            print("   ✅ FULLY COMPLIANT with SMPL-X topology")
        else:
            print("   ❌ NOT COMPLIANT with SMPL-X topology")
            for issue in compliance['issues']:
                print(f"   ⚠️  {issue}")
        
        for warning in compliance['warnings']:
            print(f"   ⚠️  {warning}")
    
    # Check consistency across files
    print(f"\n🔄 Checking mesh consistency...")
    all_files = ply_files + obj_files
    consistency = check_mesh_consistency(all_files)
    
    if consistency['consistent']:
        print("   ✅ All meshes have consistent topology")
        ref_topo = consistency['reference_topology']
        print(f"   📐 Standard topology: {ref_topo['vertices']:,} vertices, {ref_topo['faces']:,} faces")
    else:
        print("   ❌ Inconsistent topology detected:")
        for inconsistency in consistency['inconsistencies']:
            print(f"   ⚠️  {inconsistency}")
    
    # Compare with OBJ files if both exist
    if ply_files and obj_files:
        print(f"\n🆚 Comparing PLY vs OBJ format...")
        
        # Find matching files
        ply_base = ply_files[0].stem.replace('.ply', '')
        matching_obj = None
        for obj_file in obj_files:
            if obj_file.stem.replace('.obj', '') == ply_base:
                matching_obj = obj_file
                break
        
        if matching_obj:
            print(f"   Comparing: {ply_files[0].name} vs {matching_obj.name}")
            
            # Read both files
            ply_vertices, ply_faces, ply_v, ply_f = read_ply_file(ply_files[0])
            obj_vertices, obj_faces, obj_v, obj_f = read_obj_file(matching_obj)
            
            if ply_v == obj_v and ply_f == obj_f:
                print("   ✅ PLY and OBJ have identical topology")
            else:
                print(f"   ❌ Format mismatch: PLY({ply_v}v, {ply_f}f) vs OBJ({obj_v}v, {obj_f}f)")
    
    print("\n" + "=" * 50)
    
    # Final summary
    if ply_files:
        example_analysis = analyze_mesh_structure(*read_ply_file(ply_files[0])[:2])
        example_compliance = check_smplx_compliance(example_analysis)
        
        if example_compliance['is_smplx_compliant'] and consistency['consistent']:
            print("🎉 SUCCESS: Meshes are SMPL-X compliant with consistent topology!")
        elif example_compliance['is_smplx_compliant']:
            print("⚠️  PARTIAL: SMPL-X compliant but some inconsistencies detected")
        else:
            print("❌ ISSUES: Meshes do not meet SMPL-X topology standards")
            print("   Check the issues listed above for details")

if __name__ == "__main__":
    main()