#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Set environment variables for Mac compatibility  
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import smplx

def test_explicit_betas():
    print("🔍 Testing SMPL-X with explicit num_betas=10...")
    
    try:
        # Create SMPL-X model with explicit 10 betas to match our model files
        model_path = '/Users/vincent.lordier/Work/SMPLest-X/human_models/human_model_files'
        layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_jaw_pose': False, 'create_leye_pose': False, 'create_reye_pose': False, 'create_betas': False, 'create_expression': False, 'create_transl': False}
        
        smplx_model = smplx.create(model_path, 'smplx', gender='NEUTRAL', 
                                 use_pca=False, use_face_contour=True,
                                 num_betas=10, num_expression_coeffs=10, **layer_arg)
        
        print("✅ SMPL-X model created with explicit num_betas=10")
        print(f"   Shapedirs shape: {smplx_model.shapedirs.shape}")
        
        # Test tensor creation and forward pass
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        smplx_model = smplx_model.to(device)
        
        batch_size = 1
        shape = torch.randn(batch_size, 10).to(device)
        expr = torch.randn(batch_size, 10).to(device)
        
        # Create dummy poses
        body_pose = torch.zeros(batch_size, 63).to(device)
        global_orient = torch.zeros(batch_size, 3).to(device)
        left_hand_pose = torch.zeros(batch_size, 45).to(device)
        right_hand_pose = torch.zeros(batch_size, 45).to(device)
        jaw_pose = torch.zeros(batch_size, 3).to(device)
        leye_pose = torch.zeros(batch_size, 3).to(device)
        reye_pose = torch.zeros(batch_size, 3).to(device)
        
        print(f"   Input shapes - betas: {shape.shape}, expression: {expr.shape}")
        
        output = smplx_model(betas=shape, body_pose=body_pose, global_orient=global_orient,
                           left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
                           jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose,
                           expression=expr)
        
        print("✅ Forward pass successful with explicit num_betas=10")
        print(f"   Output vertices shape: {output.vertices.shape}")
        print(f"   Output joints shape: {output.joints.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_explicit_betas()
    print(f"\n🎯 Explicit betas approach: {'SUCCESS' if success else 'FAILED'}")