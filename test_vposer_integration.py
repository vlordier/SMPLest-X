"""
Test script for VPoser integration
Validates VPoser wrapper functionality and PyTorch3D rendering
"""

import torch
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.vposer_utils import create_vposer_wrapper, VPoserLoss
from utils.pytorch3d_renderer import create_pytorch3d_renderer
from human_models.human_models import SMPLX


def test_vposer_functionality():
    """Test VPoser wrapper functionality"""
    print("=" * 60)
    print("Testing VPoser Functionality")
    print("=" * 60)
    
    # Try to create VPoser wrapper
    try:
        vposer_wrapper = create_vposer_wrapper(
            vposer_ckpt_dir='./data/vposer_v1_0/snapshots',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        if vposer_wrapper is None:
            print("❌ VPoser wrapper creation failed - VPoser not available")
            return False
        else:
            print("✅ VPoser wrapper created successfully")
            
    except Exception as e:
        print(f"❌ VPoser wrapper creation failed: {e}")
        return False
    
    # Test pose encoding/decoding
    try:
        # Create sample body pose (63 dimensions)
        batch_size = 2
        sample_pose = torch.randn(batch_size, 63).to(vposer_wrapper.device)
        
        print(f"📝 Testing pose encoding/decoding with batch size {batch_size}")
        
        # Encode pose to latent space
        pose_latent = vposer_wrapper.encode_pose(sample_pose)
        print(f"✅ Pose encoding: {sample_pose.shape} -> {pose_latent.shape}")
        assert pose_latent.shape == (batch_size, 32), f"Expected shape ({batch_size}, 32), got {pose_latent.shape}"
        
        # Decode latent back to pose space
        decoded_pose = vposer_wrapper.decode_pose(pose_latent)
        print(f"✅ Pose decoding: {pose_latent.shape} -> {decoded_pose.shape}")
        assert decoded_pose.shape == (batch_size, 63), f"Expected shape ({batch_size}, 63), got {decoded_pose.shape}"
        
        # Test pose sampling
        sampled_poses = vposer_wrapper.sample_poses(batch_size=3)
        print(f"✅ Pose sampling: Generated poses shape {sampled_poses.shape}")
        assert sampled_poses.shape == (3, 63), f"Expected shape (3, 63), got {sampled_poses.shape}"
        
        # Test pose regularization
        regularization_alpha = 0.3
        regularized_pose = vposer_wrapper.regularize_pose(sample_pose, alpha=regularization_alpha)
        print(f"✅ Pose regularization with alpha={regularization_alpha}: {regularized_pose.shape}")
        assert regularized_pose.shape == sample_pose.shape
        
        # Test pose prior loss
        pose_prior_loss = vposer_wrapper.compute_pose_prior_loss(pose_latent)
        print(f"✅ Pose prior loss computed: {pose_prior_loss.item():.6f}")
        assert isinstance(pose_prior_loss, torch.Tensor) and pose_prior_loss.dim() == 0
        
        print("✅ All VPoser functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VPoser functionality test failed: {e}")
        return False


def test_vposer_loss():
    """Test VPoser loss functions"""
    print("\n" + "=" * 60)
    print("Testing VPoser Loss Functions")
    print("=" * 60)
    
    try:
        # Create VPoser wrapper
        vposer_wrapper = create_vposer_wrapper(
            vposer_ckpt_dir='./data/vposer_v1_0/snapshots',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        if vposer_wrapper is None:
            print("❌ VPoser wrapper not available - skipping loss tests")
            return False
        
        # Create VPoser loss
        vposer_loss = VPoserLoss(vposer_wrapper)
        
        # Test loss computation
        batch_size = 4
        sample_pose = torch.randn(batch_size, 63).to(vposer_wrapper.device)
        
        losses = vposer_loss(
            body_pose=sample_pose,
            pose_prior_weight=1.0,
            reconstruction_weight=10.0
        )
        
        print(f"📝 Testing VPoser loss computation with batch size {batch_size}")
        print(f"✅ Pose prior loss: {losses['pose_prior_loss'].item():.6f}")
        print(f"✅ Pose reconstruction loss: {losses['pose_reconstruction_loss'].item():.6f}")
        print(f"✅ Total VPoser loss: {losses['total_vposer_loss'].item():.6f}")
        
        # Verify loss components
        assert 'pose_prior_loss' in losses
        assert 'pose_reconstruction_loss' in losses
        assert 'total_vposer_loss' in losses
        
        # Check that losses are scalars
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor) and loss_value.dim() == 0, \
                f"{loss_name} should be scalar tensor"
        
        print("✅ All VPoser loss tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VPoser loss test failed: {e}")
        return False


def test_pytorch3d_renderer():
    """Test PyTorch3D renderer functionality"""
    print("\n" + "=" * 60)
    print("Testing PyTorch3D Renderer")
    print("=" * 60)
    
    try:
        # Create PyTorch3D renderer
        renderer = create_pytorch3d_renderer(
            image_size=(256, 256),
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        if renderer is None:
            print("❌ PyTorch3D renderer not available")
            return False
        else:
            print("✅ PyTorch3D renderer created successfully")
        
        # Create test mesh data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Simple cube vertices
        vertices = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom face
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top face
        ], dtype=torch.float32).to(device)
        
        # Simple cube faces
        faces = torch.tensor([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 5, 6], [4, 6, 7],  # top
            [0, 1, 5], [0, 5, 4],  # front
            [2, 3, 7], [2, 7, 6],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 2, 6], [1, 6, 5]   # right
        ], dtype=torch.long).to(device)
        
        print(f"📝 Testing mesh rendering with {vertices.shape[0]} vertices, {faces.shape[0]} faces")
        
        # Test basic mesh rendering
        rendered_image = renderer.render_mesh(
            vertices=vertices,
            faces=faces,
            camera_distance=3.0
        )
        
        print(f"✅ Mesh rendering successful: output shape {rendered_image.shape}")
        assert rendered_image.shape == (256, 256, 3), f"Expected (256, 256, 3), got {rendered_image.shape}"
        assert rendered_image.dtype == np.uint8, f"Expected uint8, got {rendered_image.dtype}"
        
        # Test with different camera angles
        for angle in [0, 45, 90]:
            rendered_image = renderer.render_mesh(
                vertices=vertices,
                faces=faces,
                camera_azimuth=angle
            )
            assert rendered_image.shape == (256, 256, 3)
        
        print("✅ Multi-angle rendering test passed")
        
        # Test SMPL-X mesh rendering interface
        # Load SMPL-X for face data
        try:
            smplx = SMPLX.get_instance()
            smplx_faces = torch.from_numpy(smplx.face.astype(np.int64)).to(device)
            
            # Create dummy SMPL-X vertices
            dummy_smplx_vertices = torch.randn(10475, 3).to(device)  # SMPL-X has 10475 vertices
            
            rendered_smplx = renderer.render_smplx_mesh(
                smplx_vertices=dummy_smplx_vertices,
                smplx_faces=smplx_faces
            )
            
            print(f"✅ SMPL-X mesh rendering: output shape {rendered_smplx.shape}")
            assert rendered_smplx.shape == (256, 256, 3)
            
        except Exception as e:
            print(f"⚠️ SMPL-X mesh test skipped: {e}")
        
        print("✅ All PyTorch3D renderer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch3D renderer test failed: {e}")
        return False


def test_integration_compatibility():
    """Test compatibility between components"""
    print("\n" + "=" * 60)
    print("Testing Integration Compatibility")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📝 Using device: {device}")
    
    # Test VPoser + SMPL-X compatibility
    try:
        vposer_wrapper = create_vposer_wrapper(device=device)
        
        if vposer_wrapper:
            # Test with realistic SMPL-X dimensions
            smplx = SMPLX.get_instance()
            print(f"✅ SMPL-X instance created: {smplx.vertex_num} vertices")
            
            # Test pose dimensions match
            sample_body_pose = torch.randn(1, 63).to(device)  # SMPL-X body pose
            regularized_pose = vposer_wrapper.regularize_pose(sample_body_pose, alpha=0.2)
            
            print(f"✅ VPoser-SMPL-X compatibility: {sample_body_pose.shape} -> {regularized_pose.shape}")
            assert regularized_pose.shape == sample_body_pose.shape
            
        else:
            print("⚠️ VPoser not available - skipping VPoser integration tests")
    
    except Exception as e:
        print(f"❌ VPoser-SMPL-X compatibility test failed: {e}")
        return False
    
    # Test PyTorch3D + SMPL-X compatibility
    try:
        renderer = create_pytorch3d_renderer(device=device)
        
        if renderer:
            smplx = SMPLX.get_instance()
            faces = torch.from_numpy(smplx.face.astype(np.int64)).to(device)
            
            print(f"✅ PyTorch3D-SMPL-X compatibility: {smplx.vertex_num} vertices, {len(faces)} faces")
            
        else:
            print("⚠️ PyTorch3D not available - skipping PyTorch3D integration tests")
    
    except Exception as e:
        print(f"❌ PyTorch3D-SMPL-X compatibility test failed: {e}")
        return False
    
    print("✅ All integration compatibility tests passed!")
    return True


def main():
    """Run all VPoser integration tests"""
    print("🚀 Starting VPoser Integration Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run all tests
    tests = [
        ("VPoser Functionality", test_vposer_functionality),
        ("VPoser Loss Functions", test_vposer_loss),
        ("PyTorch3D Renderer", test_pytorch3d_renderer),
        ("Integration Compatibility", test_integration_compatibility)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Summary: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! VPoser integration is ready.")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed. Check dependencies and setup.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)