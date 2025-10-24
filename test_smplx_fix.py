#!/usr/bin/env python3
"""
Test script to verify that the SMPL-X face indices fix is working.
This creates a simple mesh using the fixed SMPL-X model.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from human_models.human_models import SMPLX

def test_smplx_faces_fix():
    """Test that SMPL-X face indices are now properly loaded"""
    print("🔍 TESTING SMPL-X FACE INDICES FIX")
    print("=" * 50)
    
    try:
        # Initialize SMPL-X (using the path from config)
        smpl_x = SMPLX('./human_models/human_model_files')
        
        print(f"✅ SMPL-X model initialized successfully")
        print(f"   Face array shape: {smpl_x.face.shape}")
        print(f"   First 5 faces: {smpl_x.face[:5]}")
        print(f"   Last 5 faces: {smpl_x.face[-5:]}")
        
        # Check if faces are valid
        all_zeros = (smpl_x.face == 0).all()
        unique_values = len(set(smpl_x.face.flatten()))
        min_face_idx = smpl_x.face.min()
        max_face_idx = smpl_x.face.max()
        
        print(f"   Are all faces zeros? {all_zeros}")
        print(f"   Min face index: {min_face_idx}")
        print(f"   Max face index: {max_face_idx}")
        print(f"   Unique face values: {unique_values}")
        
        if all_zeros:
            print("❌ FACES STILL ALL ZEROS - FIX FAILED!")
            return False
        elif unique_values < 100:
            print("⚠️  Very few unique face indices - possible issue")
            return False
        elif max_face_idx >= 10475:
            print("⚠️  Face indices exceed vertex count - possible issue")
            return False
        else:
            print("✅ FACES LOOK GOOD - FIX SUCCESSFUL!")
            
            # Test creating a simple mesh
            print(f"\n🎯 Testing mesh generation...")
            
            # Create dummy parameters 
            batch_size = 1
            betas = torch.zeros(batch_size, 10)  # Shape parameters
            body_pose = torch.zeros(batch_size, 63)  # Body pose
            global_orient = torch.zeros(batch_size, 3)  # Global orientation
            
            # Generate mesh using one of the SMPL-X layers
            smplx_output = smpl_x.layer['neutral'](
                betas=betas,
                body_pose=body_pose,
                global_orient=global_orient,
                return_verts=True
            )
            
            vertices = smplx_output.vertices[0].detach().numpy()  # [10475, 3]
            faces = smpl_x.face  # [20908, 3]
            
            print(f"   Generated vertices shape: {vertices.shape}")
            print(f"   Generated faces shape: {faces.shape}")
            print(f"   Vertex bounds: [{vertices.min():.3f}, {vertices.max():.3f}]")
            
            # Check if vertices are all identical (the original issue)
            unique_vertices = np.unique(vertices.round(6), axis=0)
            print(f"   Unique vertices (rounded): {len(unique_vertices)}")
            
            if len(unique_vertices) == 1:
                print("❌ ALL VERTICES IDENTICAL - MESH GENERATION STILL BROKEN!")
                return False
            elif len(unique_vertices) < 100:
                print("⚠️  Very few unique vertices - possible mesh issue")
            else:
                print("✅ VERTICES HAVE PROPER VARIATION - MESH GENERATION WORKS!")
            
            # Save a test PLY file to verify
            test_ply_path = "test_fixed_mesh.ply"
            save_test_ply(vertices[:100], faces[:100], test_ply_path)  # Save sample
            print(f"   Saved test mesh sample: {test_ply_path}")
            
            return True
    
    except Exception as e:
        print(f"❌ Error testing SMPL-X fix: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_test_ply(vertices, faces, filepath):
    """Save a simple PLY file for testing"""
    with open(filepath, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write(f'element face {len(faces)}\n')
        f.write('property list uchar int vertex_indices\n')
        f.write('end_header\n')
        
        # Write vertices
        for vertex in vertices:
            f.write(f'{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        
        # Write faces (only those that reference vertices we included)
        valid_faces = []
        for face in faces:
            if all(idx < len(vertices) for idx in face):
                valid_faces.append(face)
                if len(valid_faces) >= 50:  # Limit sample size
                    break
        
        for face in valid_faces:
            f.write(f'3 {face[0]} {face[1]} {face[2]}\n')

if __name__ == "__main__":
    success = test_smplx_faces_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SMPL-X FACE FIX VERIFICATION: SUCCESS!")
        print("   The SMPL-X model files have been successfully repaired.")
        print("   Face indices are now properly loaded and mesh generation works.")
    else:
        print("❌ SMPL-X FACE FIX VERIFICATION: FAILED!")
        print("   The fix did not work or there are still issues.")