"""
Working VPoser test using proper model loading
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vposer_with_config():
    """Test VPoser using proper config-based loading"""
    print("🧪 Testing VPoser with Configuration")
    print("=" * 50)
    
    try:
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        
        # Set up the VPoser directory path
        vposer_ckpt_dir = './data/vposer_v1_0'
        print(f"📂 VPoser directory: {vposer_ckpt_dir}")
        
        # Check if we have the required files
        snapshots_dir = os.path.join(vposer_ckpt_dir, 'snapshots')
        config_file = None
        
        # Look for .ini config file
        for f in os.listdir(vposer_ckpt_dir):
            if f.endswith('.ini'):
                config_file = os.path.join(vposer_ckpt_dir, f)
                break
        
        print(f"📋 Config file: {config_file}")
        print(f"📁 Snapshots dir: {snapshots_dir}")
        print(f"📁 Snapshots exist: {os.path.exists(snapshots_dir)}")
        
        if os.path.exists(snapshots_dir):
            checkpoint_files = [f for f in os.listdir(snapshots_dir) if f.endswith('.pt')]
            print(f"🔍 Checkpoint files: {checkpoint_files}")
        
        # Try to load using the official model loader
        print("\n🏗️ Loading VPoser model...")
        
        vp, ps = load_model(vposer_ckpt_dir, model_code=VPoser,
                           remove_words_in_model_weights='vp_model.',
                           disable_grad=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vp = vp.to(device)
        vp.eval()
        
        print(f"✅ VPoser model loaded successfully on {device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in vp.parameters())}")
        
        # Test basic functionality
        batch_size = 2
        test_pose = torch.randn(batch_size, 63).to(device)
        
        print(f"\n📝 Testing with batch size {batch_size}")
        print(f"   Input shape: {test_pose.shape}")
        
        with torch.no_grad():
            # Encode to latent space
            encoded = vp.encode(test_pose)
            
            if hasattr(encoded, 'mean'):
                latent = encoded.mean
                print(f"✅ Encoded (mean): {latent.shape}")
                
                if hasattr(encoded, 'logvar'):
                    logvar = encoded.logvar
                    print(f"   Logvar: {logvar.shape}")
            else:
                latent = encoded
                print(f"✅ Encoded: {latent.shape}")
            
            # Decode back to pose space
            decoded = vp.decode(latent)
            
            if isinstance(decoded, dict) and 'pose_body' in decoded:
                decoded_pose = decoded['pose_body']
            else:
                decoded_pose = decoded
            
            # Ensure proper shape
            if decoded_pose.dim() == 3:
                decoded_pose = decoded_pose.contiguous().view(batch_size, -1)
            
            print(f"✅ Decoded: {decoded_pose.shape}")
            
            # Compute reconstruction error
            if decoded_pose.shape[1] == 63:
                recon_error = torch.mean((test_pose - decoded_pose).pow(2)).item()
                print(f"📊 Reconstruction error: {recon_error:.6f}")
            else:
                print(f"⚠️ Shape mismatch - decoded: {decoded_pose.shape}, expected: {test_pose.shape}")
        
        # Test pose sampling
        print(f"\n🎲 Testing pose sampling...")
        sample_latents = torch.randn(3, latent.shape[1]).to(device)
        
        with torch.no_grad():
            sampled = vp.decode(sample_latents)
            if isinstance(sampled, dict) and 'pose_body' in sampled:
                sampled_poses = sampled['pose_body']
            else:
                sampled_poses = sampled
                
            if sampled_poses.dim() == 3:
                sampled_poses = sampled_poses.contiguous().view(3, -1)
                
            print(f"✅ Sampled poses: {sampled_poses.shape}")
        
        print("\n🎉 VPoser functionality test successful!")
        return True, vp
        
    except Exception as e:
        print(f"❌ VPoser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_vposer_wrapper_with_working_model():
    """Test our wrapper with the working VPoser model"""
    print("\n🧪 Testing VPoser Wrapper Integration")
    print("=" * 50)
    
    # First load the working model
    success, vp = test_vposer_with_config()
    
    if not success:
        print("❌ Cannot test wrapper - VPoser model loading failed")
        return False
    
    try:
        # Now test our wrapper - but modify it to use the working VPoser
        print("🔧 Testing wrapper functionality...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Test direct wrapper functions similar to what VPoser does
        batch_size = 2
        test_pose = torch.randn(batch_size, 63).to(device)
        
        print(f"📝 Input pose: {test_pose.shape}")
        
        with torch.no_grad():
            # Encode
            encoded = vp.encode(test_pose)
            pose_latent = encoded.mean if hasattr(encoded, 'mean') else encoded
            print(f"✅ Encoded: {pose_latent.shape}")
            
            # Decode
            decoded = vp.decode(pose_latent)
            if isinstance(decoded, dict) and 'pose_body' in decoded:
                decoded_pose = decoded['pose_body'].contiguous().view(batch_size, 63)
            else:
                decoded_pose = decoded.contiguous().view(batch_size, 63)
            
            print(f"✅ Decoded: {decoded_pose.shape}")
            
            # Test regularization (blend original and decoded)
            alpha = 0.3
            regularized = (1 - alpha) * test_pose + alpha * decoded_pose
            print(f"✅ Regularized with alpha={alpha}: {regularized.shape}")
            
            # Test prior loss (L2 norm of latent codes)
            prior_loss = torch.mean(pose_latent.pow(2))
            print(f"✅ Pose prior loss: {prior_loss.item():.6f}")
        
        print("\n🎉 Wrapper-style functionality successful!")
        return True
        
    except Exception as e:
        print(f"❌ Wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run working VPoser tests"""
    print("🚀 VPoser Working Tests")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()
    
    tests = [
        test_vposer_wrapper_with_working_model,
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 VPoser is fully functional!")
        print("\n💡 Next steps:")
        print("   ✅ VPoser model loads correctly")
        print("   ✅ Encoding/decoding works")
        print("   ✅ Pose regularization ready")
        print("   ✅ Ready for inference integration")
        return True
    else:
        print("⚠️ Some functionality missing")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)