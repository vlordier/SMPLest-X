#!/usr/bin/env python3
"""
Debug the mesh generation issue to understand why all vertices are identical.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.SMPLest_X import get_model
from configs import config_smplest_x_h as cfg_module
from human_models.human_models import SMPLX

class ConfigObj:
    """Convert config dict to object with proper attribute access"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObj(value))
            else:
                setattr(self, key, value)

def debug_model_output():
    """Debug what the model is actually outputting"""
    
    print("🔍 DEBUGGING MESH GENERATION ISSUE")
    print("=" * 50)
    
    # Initialize model
    cfg = ConfigObj(cfg_module.config)
    cfg.model.pretrained_model_path = './pretrained_models/smplest_x_h/smplest_x_h.pth.tar'
    
    print("Loading model...")
    model = get_model(cfg, mode='test')
    model.eval()
    model = model.cpu()  # Force CPU to avoid MPS issues
    
    # Initialize SMPL-X for face indices
    smpl_x = SMPLX(cfg.model.human_model_path)
    
    print(f"SMPL-X faces shape: {smpl_x.face.shape}")
    print(f"SMPL-X faces sample: {smpl_x.face[:5]}")
    
    # Create dummy inputs (what the inference script would pass to the model)
    batch_size = 1
    inputs = {
        'img': torch.randn(batch_size, 3, 512, 384).cpu()  # Standard input size, force CPU
    }
    targets = {}
    meta_info = {}
    
    print("\nRunning model inference...")
    with torch.no_grad():
        model_output = model(inputs, targets, meta_info, 'test')
    
    print(f"Model output keys: {list(model_output.keys())}")
    
    if 'smplx_mesh_cam' in model_output:
        mesh_tensor = model_output['smplx_mesh_cam']
        print(f"\nMesh tensor info:")
        print(f"  Shape: {mesh_tensor.shape}")
        print(f"  Device: {mesh_tensor.device}")
        print(f"  Dtype: {mesh_tensor.dtype}")
        print(f"  Min/Max: [{mesh_tensor.min().item():.6f}, {mesh_tensor.max().item():.6f}]")
        print(f"  Mean: {mesh_tensor.mean().item():.6f}")
        
        # Check if all vertices are the same
        mesh_vertices = mesh_tensor[0].detach().cpu().numpy()
        print(f"\nMesh vertices (numpy):")
        print(f"  Shape: {mesh_vertices.shape}")
        print(f"  First vertex: {mesh_vertices[0]}")
        print(f"  Last vertex: {mesh_vertices[-1]}")
        print(f"  Min vertex: {mesh_vertices.min(axis=0)}")
        print(f"  Max vertex: {mesh_vertices.max(axis=0)}")
        
        # Check if vertices are identical
        unique_vertices = np.unique(mesh_vertices, axis=0)
        print(f"  Unique vertices: {len(unique_vertices)}")
        
        if len(unique_vertices) == 1:
            print("  ❌ ALL VERTICES ARE IDENTICAL!")
            print(f"  All vertices equal: {unique_vertices[0]}")
        elif len(unique_vertices) < 10:
            print("  ⚠️  Very few unique vertices detected")
            for i, vertex in enumerate(unique_vertices):
                print(f"    Unique {i+1}: {vertex}")
        else:
            print("  ✅ Vertices have proper variation")
            
        # Check a few individual vertices
        print(f"\nSample vertices:")
        for i in [0, 100, 1000, 5000, 10000]:
            if i < len(mesh_vertices):
                print(f"  Vertex {i}: {mesh_vertices[i]}")
        
        # Check if this matches what gets saved to file
        mesh_faces = smpl_x.face
        print(f"\nFaces info:")
        print(f"  Shape: {mesh_faces.shape}")
        print(f"  Sample faces: {mesh_faces[:3]}")
        
        # Test saving
        print(f"\nTesting mesh save...")
        test_ply_path = "debug_test_mesh.ply"
        
        try:
            with open(test_ply_path, 'w') as f:
                f.write('ply\n')
                f.write('format ascii 1.0\n')
                f.write(f'element vertex {len(mesh_vertices)}\n')
                f.write('property float x\n')
                f.write('property float y\n')
                f.write('property float z\n')
                f.write(f'element face {len(mesh_faces)}\n')
                f.write('property list uchar int vertex_indices\n')
                f.write('end_header\n')
                
                # Write first 10 vertices
                for i in range(min(10, len(mesh_vertices))):
                    v = mesh_vertices[i]
                    f.write(f'{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                
                print(f"  Saved first 10 vertices to {test_ply_path}")
                print(f"  Check the file to see if vertices are different")
                
        except Exception as e:
            print(f"  Error saving test file: {e}")
            
    else:
        print("❌ No 'smplx_mesh_cam' in model output!")
        
    print(f"\n🔍 Model output tensor inspection:")
    for key, value in model_output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}, range [{value.min().item():.3f}, {value.max().item():.3f}]")

def main():
    debug_model_output()

if __name__ == "__main__":
    main()