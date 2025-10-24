"""
Test VPoser functionality with available model weights
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vposer_basic():
    """Test basic VPoser functionality"""
    print("🧪 Testing Basic VPoser Functionality")
    print("=" * 50)
    
    try:
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        
        # Load VPoser model
        vposer_ckpt_dir = './data/vposer_v1_0'
        print(f"📂 Loading VPoser from: {vposer_ckpt_dir}")
        
        vp, ps = load_model(vposer_ckpt_dir, model_code=VPoser,
                           remove_words_in_model_weights='vp_model.',
                           disable_grad=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vp = vp.to(device)
        vp.eval()
        
        print(f"✅ VPoser model loaded successfully on {device}")
        
        # Test encoding/decoding
        batch_size = 2
        sample_pose = torch.randn(batch_size, 63).to(device)
        
        print(f"📝 Testing with batch size {batch_size}")
        print(f"   Input pose shape: {sample_pose.shape}")
        
        # Encode to latent space
        with torch.no_grad():
            pose_latent = vp.encode(sample_pose).mean
        
        print(f"✅ Encoded to latent space: {pose_latent.shape}")
        assert pose_latent.shape == (batch_size, 32), f"Expected (2, 32), got {pose_latent.shape}"
        
        # Decode back
        with torch.no_grad():
            decoded = vp.decode(pose_latent)
            decoded_pose = decoded['pose_body'].contiguous().view(-1, 63)
        
        print(f"✅ Decoded back to pose: {decoded_pose.shape}")
        assert decoded_pose.shape == (batch_size, 63), f"Expected (2, 63), got {decoded_pose.shape}"
        
        # Check reconstruction error
        recon_error = torch.mean((sample_pose - decoded_pose).pow(2)).item()
        print(f"📊 Reconstruction error: {recon_error:.6f}")
        
        # Test pose sampling
        sample_latents = torch.randn(3, 32).to(device)
        with torch.no_grad():
            sampled_poses = vp.decode(sample_latents)['pose_body'].contiguous().view(-1, 63)
        
        print(f"✅ Pose sampling: {sampled_poses.shape}")
        assert sampled_poses.shape == (3, 63)
        
        print("\n🎉 All VPoser tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VPoser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vposer_wrapper():
    """Test our VPoser wrapper"""
    print("\n🧪 Testing VPoser Wrapper")
    print("=" * 50)
    
    try:
        from utils.vposer_utils import create_vposer_wrapper
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vposer_wrapper = create_vposer_wrapper(
            vposer_ckpt_dir='./data/vposer_v1_0',
            device=device
        )
        
        if vposer_wrapper is None:
            print("❌ VPoser wrapper creation failed")
            return False
        
        print("✅ VPoser wrapper created successfully")
        
        # Test wrapper functions
        batch_size = 2
        sample_pose = torch.randn(batch_size, 63).to(device)
        
        # Test encoding
        pose_latent = vposer_wrapper.encode_pose(sample_pose)
        print(f"✅ Wrapper encoding: {sample_pose.shape} -> {pose_latent.shape}")
        
        # Test decoding
        decoded_pose = vposer_wrapper.decode_pose(pose_latent)
        print(f"✅ Wrapper decoding: {pose_latent.shape} -> {decoded_pose.shape}")
        
        # Test regularization
        alpha = 0.3
        regularized_pose = vposer_wrapper.regularize_pose(sample_pose, alpha=alpha)
        print(f"✅ Pose regularization with alpha={alpha}: {regularized_pose.shape}")
        
        # Test pose prior loss
        pose_prior_loss = vposer_wrapper.compute_pose_prior_loss(pose_latent)
        print(f"✅ Pose prior loss: {pose_prior_loss.item():.6f}")
        
        # Test pose sampling
        sampled_poses = vposer_wrapper.sample_poses(batch_size=3)
        print(f"✅ Pose sampling: {sampled_poses.shape}")
        
        print("\n🎉 VPoser wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ VPoser wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run VPoser functionality tests"""
    print("🚀 VPoser Functionality Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    tests_passed = 0
    total_tests = 0
    
    # Test basic VPoser functionality
    total_tests += 1
    if test_vposer_basic():
        tests_passed += 1
    
    # Test our wrapper
    total_tests += 1
    if test_vposer_wrapper():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print(f"🎯 Final Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 VPoser integration is fully functional!")
        print("\n💡 Next steps:")
        print("   - Ready to run VPoser-enhanced inference")
        print("   - Can proceed with pose regularization")
        return True
    else:
        print("⚠️ Some tests failed - check setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)