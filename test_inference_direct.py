#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

# Set environment variables for Mac compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import torch
import numpy as np
from models.SMPLest_X import get_model

class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.model = MockModelConfig()
        self.data = MockDataConfig()

class MockModelConfig:
    def __init__(self):
        self.human_model_path = '/Users/vincent.lordier/Work/SMPLest-X/human_models/human_model_files'
        self.focal = [1000.0, 1000.0]
        self.princpt = [256.0, 256.0]
        self.camera_3d_size = 2.5
        self.input_body_shape = [512, 512]
        self.output_hm_shape = [64, 64, 64]

class MockDataConfig:
    def __init__(self):
        self.testset = 'demo'

def test_smplest_x_inference():
    print("🔍 Testing SMPLest-X inference with Direct SMPL-X...")
    
    try:
        # Create mock configuration
        config = MockConfig()
        
        # Create model (this will test our Direct SMPL-X implementation)
        model = get_model(config, 'test')
        print("✅ SMPLest-X model created successfully")
        
        # Move model to device
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy input data
        batch_size = 1
        inputs = {
            'img': torch.randn(batch_size, 3, 512, 512).to(device)
        }
        
        # Create dummy targets (not used in inference)
        targets = {}
        
        # Create dummy meta_info
        meta_info = {}
        
        print(f"   Input image shape: {inputs['img'].shape}")
        print(f"   Device: {device}")
        
        # Test inference forward pass
        with torch.no_grad():
            output = model(inputs, targets, meta_info, mode='test')
        
        print("✅ Inference forward pass successful")
        
        # Check outputs
        expected_keys = ['img', 'smplx_joint_proj', 'smplx_mesh_cam', 'smplx_root_pose', 
                        'smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose', 
                        'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans', 'smplx_joint_cam']
        
        for key in expected_keys:
            if key in output:
                if hasattr(output[key], 'shape'):
                    print(f"   {key}: {output[key].shape}")
                else:
                    print(f"   {key}: {type(output[key])}")
        
        # Test specific outputs
        if 'smplx_mesh_cam' in output:
            print(f"✅ Mesh vertices generated: {output['smplx_mesh_cam'].shape}")
        if 'smplx_joint_proj' in output:
            print(f"✅ Joint projections generated: {output['smplx_joint_proj'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ SMPLest-X inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_smplest_x_inference()
    print(f"\n🎯 SMPLest-X with Direct SMPL-X: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        print("\n🎉 SUCCESS: The Direct SMPL-X implementation has resolved the coefficient mismatch issue!")
        print("The SMPLest-X model can now run inference without einsum broadcast errors.")
    else:
        print("\n💥 The Direct SMPL-X implementation still has issues that need to be resolved.")