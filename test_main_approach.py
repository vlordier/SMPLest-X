#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Set environment variables for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
from human_models.human_models import SMPLX

def test_main_approach():
    print("🔍 Testing main branch approach...")
    
    # Test SMPL-X model creation
    try:
        smplx_model = SMPLX('/Users/vincent.lordier/Work/SMPLest-X/human_models/human_model_files')
        print("✅ SMPL-X model created successfully")
        print(f"   Shape param dim: {smplx_model.shape_param_dim}")
        print(f"   Expression dim: {smplx_model.expr_code_dim}")
        print(f"   Shapedirs shape: {smplx_model.layer['neutral'].shapedirs.shape}")
        
        # Test tensor creation and forward pass
        batch_size = 1
        shape = torch.randn(batch_size, 10)  # 10 shape coefficients
        expr = torch.randn(batch_size, 10)   # 10 expression coefficients
        
        print(f"   Shape tensor: {shape.shape}")
        print(f"   Expression tensor: {expr.shape}")
        
        # Test if direct forward pass works
        try:
            layer = smplx_model.layer['neutral']
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            layer = layer.to(device)
            shape = shape.to(device)
            expr = expr.to(device)
            
            # Create dummy poses
            body_pose = torch.zeros(batch_size, 63).to(device)
            global_orient = torch.zeros(batch_size, 3).to(device)
            left_hand_pose = torch.zeros(batch_size, 45).to(device)
            right_hand_pose = torch.zeros(batch_size, 45).to(device)
            jaw_pose = torch.zeros(batch_size, 3).to(device)
            leye_pose = torch.zeros(batch_size, 3).to(device)
            reye_pose = torch.zeros(batch_size, 3).to(device)
            
            output = layer(betas=shape, body_pose=body_pose, global_orient=global_orient,
                          left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                          jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
                          expression=expr)
            
            print("✅ Forward pass successful with main branch approach")
            print(f"   Output vertices shape: {output.vertices.shape}")
            print(f"   Output joints shape: {output.joints.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ SMPL-X model creation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_main_approach()
    print(f"\n🎯 Main branch approach: {'SUCCESS' if success else 'FAILED'}")