#!/usr/bin/env python3
"""
Quick test to verify SMPL-X integration and mesh generation
"""

import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_smplx_loading():
    """Test SMPL-X model loading"""
    print("🔧 Testing SMPL-X model loading...")
    
    try:
        # Import SMPL-X from the correct location
        from human_models.human_models import SMPLX
        
        # Initialize SMPL-X model
        smplx_model_path = './human_models/human_model_files'
        smplx = SMPLX(smplx_model_path)
        
        print("✅ SMPL-X model loaded successfully")
        print(f"  📊 Number of vertices: {smplx.vertex_num}")
        print(f"  📊 Number of faces: {smplx.face.shape[0]}")
        
        return smplx
        
    except Exception as e:
        print(f"❌ SMPL-X loading failed: {e}")
        print("  ℹ️ This is expected if human_models package is not properly configured")
        return None

def test_mesh_generation(smplx_model):
    """Test mesh generation with SMPL-X"""
    if smplx_model is None:
        return False
        
    print("🔧 Testing mesh generation...")
    
    try:
        # Create test parameters
        batch_size = 1
        global_orient = torch.zeros(batch_size, 3)
        body_pose = torch.randn(batch_size, 63) * 0.1  # Small random pose
        left_hand_pose = torch.zeros(batch_size, 45)
        right_hand_pose = torch.zeros(batch_size, 45)
        jaw_pose = torch.zeros(batch_size, 3)
        leye_pose = torch.zeros(batch_size, 3)
        reye_pose = torch.zeros(batch_size, 3)
        betas = torch.zeros(batch_size, 10)
        expression = torch.zeros(batch_size, 10)
        transl = torch.zeros(batch_size, 3)
        
        # Generate mesh using the SMPL-X layer
        with torch.no_grad():
            smplx_output = smplx_model.layer['neutral'](
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                betas=betas,
                expression=expression,
                transl=transl
            )
        
        vertices = smplx_output.vertices[0]
        print(f"✅ Mesh generated successfully: {vertices.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Mesh generation failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 SMPL-X Integration Test")
    print("=" * 50)
    
    # Test SMPL-X loading
    smplx_model = test_smplx_loading()
    
    print("\n" + "=" * 50)
    
    # Test mesh generation
    if smplx_model is not None:
        mesh_success = test_mesh_generation(smplx_model)
    else:
        mesh_success = False
    
    print("\n" + "=" * 50)
    print("📋 Test Results")
    print("=" * 50)
    
    if smplx_model is not None:
        print("✅ SMPL-X model: Available and functional")
        if mesh_success:
            print("✅ Mesh generation: Working")
            print("🎉 SMPL-X integration is ready for mesh visualization!")
        else:
            print("❌ Mesh generation: Failed")
    else:
        print("⚠️ SMPL-X model: Not available")
        print("ℹ️ VPoser integration works without mesh visualization")
    
    return smplx_model is not None and mesh_success

if __name__ == "__main__":
    success = main()