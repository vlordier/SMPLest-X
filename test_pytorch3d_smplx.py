#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Set environment variables for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
from human_models.pytorch3d_smplx import Direct_SMPLX
from models.SMPLest_X import get_model

def test_pytorch3d_smplx():
    print("🔍 Testing Direct SMPL-X implementation...")
    
    try:
        # Test Direct SMPL-X model creation
        smplx_model = Direct_SMPLX('/Users/vincent.lordier/Work/SMPLest-X/human_models/human_model_files')
        print("✅ Direct SMPL-X model created successfully")
        print(f"   Shape param dim: {smplx_model.shape_param_dim}")
        print(f"   Expression dim: {smplx_model.expr_code_dim}")
        print(f"   Vertex num: {smplx_model.vertex_num}")
        
        # Test tensor creation and forward pass
        batch_size = 1
        device = smplx_model.device
        
        shape = torch.randn(batch_size, 10).to(device)  # 10 shape coefficients
        expr = torch.randn(batch_size, 10).to(device)   # 10 expression coefficients
        
        print(f"   Shape tensor: {shape.shape}")
        print(f"   Expression tensor: {expr.shape}")
        
        # Create dummy poses
        body_pose = torch.zeros(batch_size, 63).to(device)
        global_orient = torch.zeros(batch_size, 3).to(device)
        left_hand_pose = torch.zeros(batch_size, 45).to(device)
        right_hand_pose = torch.zeros(batch_size, 45).to(device)
        jaw_pose = torch.zeros(batch_size, 3).to(device)
        leye_pose = torch.zeros(batch_size, 3).to(device)
        reye_pose = torch.zeros(batch_size, 3).to(device)
        transl = torch.zeros(batch_size, 3).to(device)
        
        # Test forward pass
        output = smplx_model.forward(
            betas=shape, body_pose=body_pose, global_orient=global_orient,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
            expression=expr, transl=transl
        )
        
        print("✅ Forward pass successful with Direct SMPL-X approach")
        print(f"   Output vertices shape: {output.vertices.shape}")
        print(f"   Output joints shape: {output.joints.shape}")
        
        # Test compatibility layer
        layer = smplx_model.layer['neutral']
        output2 = layer(
            betas=shape, body_pose=body_pose, global_orient=global_orient,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
            expression=expr, transl=transl
        )
        
        print("✅ Compatibility layer works")
        print(f"   Faces shape: {layer.faces.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct SMPL-X test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pytorch3d_smplx()
    print(f"\n🎯 Direct SMPL-X approach: {'SUCCESS' if success else 'FAILED'}")