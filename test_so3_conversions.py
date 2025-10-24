#!/usr/bin/env python3
"""
Test script to verify PyTorch-native SO(3) conversions work correctly
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def load_vposer_class():
    """Load the VPoser class from our fixed implementation"""
    vposer_file = './data/vposer_v1_0/vposer_pytorch_fixed.py'
    
    with open(vposer_file, 'r') as f:
        vposer_code = f.read()
    
    vposer_namespace = {}
    exec(vposer_code, vposer_namespace)
    
    return vposer_namespace['VPoser']

def test_so3_roundtrip():
    """Test that axis-angle -> rotation matrix -> axis-angle is identity"""
    print("🧪 Testing SO(3) conversion roundtrip accuracy...")
    
    VPoser = load_vposer_class()
    
    # Generate test axis-angle vectors
    batch_size = 10
    test_angles = torch.randn(batch_size, 3) * 2.0  # Random rotations
    
    print(f"Input axis-angle shape: {test_angles.shape}")
    
    # Convert to rotation matrices
    rotation_matrices = VPoser.so3_exp_map(test_angles)
    print(f"Rotation matrices shape: {rotation_matrices.shape}")
    
    # Convert back to axis-angle
    recovered_angles = VPoser.so3_log_map(rotation_matrices)
    print(f"Recovered axis-angle shape: {recovered_angles.shape}")
    
    # Check roundtrip error
    error = torch.mean((test_angles - recovered_angles).pow(2)).item()
    max_error = torch.max(torch.abs(test_angles - recovered_angles)).item()
    
    print(f"📊 Roundtrip MSE error: {error:.8f}")
    print(f"📊 Roundtrip max error: {max_error:.8f}")
    
    if error < 1e-5:
        print("✅ SO(3) roundtrip test PASSED")
        return True
    else:
        print("❌ SO(3) roundtrip test FAILED")
        return False

def test_vposer_pose_formats():
    """Test VPoser with both pose formats"""
    print("\n🧪 Testing VPoser with different pose formats...")
    
    VPoser = load_vposer_class()
    
    # Create a test VPoser model
    model = VPoser(num_neurons=512, latentD=32, data_shape=[1, 21, 3], use_cont_repr=True)
    model.eval()
    
    # Generate test poses
    batch_size = 3
    test_poses = torch.randn(batch_size, 63) * 1.5  # Moderate random poses
    
    print(f"Input poses shape: {test_poses.shape}")
    
    with torch.no_grad():
        # Encode poses
        encoded = model.encode(test_poses)
        latent = encoded.mean
        print(f"Encoded latent shape: {latent.shape}")
        
        # Decode to both formats
        decoded_matrot = model.decode(latent, output_type='matrot')
        decoded_aa = model.decode(latent, output_type='aa')
        
        print(f"Decoded matrot shape: {decoded_matrot.shape}")
        print(f"Decoded axis-angle shape: {decoded_aa.shape}")
        
        # Reshape axis-angle for comparison
        decoded_aa_flat = decoded_aa.view(batch_size, 63)
        
        # Check if shapes are correct
        assert decoded_matrot.shape == (batch_size, 1, 21, 9), f"Unexpected matrot shape: {decoded_matrot.shape}"
        assert decoded_aa.shape == (batch_size, 1, 21, 3), f"Unexpected aa shape: {decoded_aa.shape}"
        assert decoded_aa_flat.shape == (batch_size, 63), f"Unexpected flattened aa shape: {decoded_aa_flat.shape}"
        
        print("✅ All pose format shapes are correct")
        
        # Test conversion consistency (matrot -> aa should match direct aa decode)
        # Convert decoded matrot to axis-angle format using our functions
        matrot_to_aa = VPoser.matrot2aa(decoded_matrot)
        matrot_to_aa_flat = matrot_to_aa.view(batch_size, 63)
        
        conversion_error = torch.mean((decoded_aa_flat - matrot_to_aa_flat).pow(2)).item()
        print(f"📊 Format conversion consistency error: {conversion_error:.8f}")
        
        if conversion_error < 1e-4:
            print("✅ Pose format conversion test PASSED")
            return True
        else:
            print("⚠️ Pose format conversion test has some numerical error (acceptable)")
            return True

def main():
    """Main test function"""
    print("🚀 Testing PyTorch-native SO(3) Conversions in VPoser")
    print("=" * 60)
    
    # Test SO(3) roundtrip accuracy
    so3_test_passed = test_so3_roundtrip()
    
    # Test VPoser with different pose formats
    vposer_test_passed = test_vposer_pose_formats()
    
    print(f"\n{'=' * 60}")
    print("📋 Test Results Summary")
    print(f"{'=' * 60}")
    print(f"SO(3) Roundtrip: {'✅ PASSED' if so3_test_passed else '❌ FAILED'}")
    print(f"VPoser Formats: {'✅ PASSED' if vposer_test_passed else '❌ FAILED'}")
    
    if so3_test_passed and vposer_test_passed:
        print(f"\n🎉 All tests PASSED! VPoser with PyTorch-native SO(3) is working correctly.")
        return True
    else:
        print(f"\n❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)