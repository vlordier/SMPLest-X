"""
Direct VPoser model test - bypassing the model loader
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_vposer_load():
    """Test loading VPoser checkpoint directly"""
    print("🧪 Testing Direct VPoser Model Loading")
    print("=" * 50)
    
    try:
        from human_body_prior.models.vposer_model import VPoser
        
        # Try to load the checkpoint directly
        checkpoint_path = './data/vposer_v1_0/snapshots/TR00_E096.pt'
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ Checkpoint loaded successfully")
        print(f"📋 Checkpoint keys: {list(ckpt.keys())}")
        
        # Check model state dict
        if 'model_state_dict' in ckpt:
            model_state_dict = ckpt['model_state_dict']
            print(f"📋 Model keys: {list(model_state_dict.keys())[:10]}...")  # Show first 10 keys
        elif 'state_dict' in ckpt:
            model_state_dict = ckpt['state_dict']
            print(f"📋 State dict keys: {list(model_state_dict.keys())[:10]}...")
        else:
            # The checkpoint might be the model state dict directly
            model_state_dict = ckpt
            print(f"📋 Direct model keys: {list(model_state_dict.keys())[:10]}...")
        
        # Try to create VPoser model and load weights
        # We need to figure out the architecture
        print("\n🏗️ Attempting to create VPoser model...")
        
        # VPoser typically has encoder/decoder structure
        # Let's try to infer the latent dimension from the weights
        latent_dim = 32  # Standard VPoser latent dimension
        
        # Check if we can find architecture info in the checkpoint
        if 'n_neuron' in str(model_state_dict.keys()):
            print("   Found n_neuron architecture information")
        
        print(f"   Using latent dimension: {latent_dim}")
        
        # Create VPoser model with default parameters
        vposer_model = VPoser(num_neurons=512, latentD=latent_dim, data_shape=[1, 1, 63])
        
        # Try to load the weights
        vposer_model.load_state_dict(model_state_dict, strict=False)
        print("✅ VPoser model created and weights loaded")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vposer_model = vposer_model.to(device)
        vposer_model.eval()
        
        # Test basic functionality
        batch_size = 2
        test_pose = torch.randn(batch_size, 63).to(device)
        
        print(f"📝 Testing encoding/decoding with shape {test_pose.shape}")
        
        with torch.no_grad():
            # Encode
            encoded = vposer_model.encode(test_pose)
            if hasattr(encoded, 'mean'):
                latent = encoded.mean
            else:
                latent = encoded
                
            print(f"✅ Encoded to latent: {latent.shape}")
            
            # Decode
            decoded = vposer_model.decode(latent)
            if isinstance(decoded, dict) and 'pose_body' in decoded:
                decoded_pose = decoded['pose_body']
            else:
                decoded_pose = decoded
                
            decoded_pose = decoded_pose.contiguous().view(batch_size, 63)
            print(f"✅ Decoded to pose: {decoded_pose.shape}")
            
            # Compute reconstruction error
            recon_error = torch.mean((test_pose - decoded_pose).pow(2)).item()
            print(f"📊 Reconstruction error: {recon_error:.6f}")
        
        print("\n🎉 Direct VPoser loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Direct VPoser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_architecture_inference():
    """Try to infer VPoser model architecture from checkpoint"""
    print("\n🔍 Inferring VPoser Model Architecture")
    print("=" * 50)
    
    try:
        checkpoint_path = './data/vposer_v1_0/snapshots/TR00_E096.pt'
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Get model state dict
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        
        print("🔍 Analyzing model architecture from weights:")
        
        # Look for encoder/decoder patterns
        encoder_keys = [k for k in state_dict.keys() if 'encoder' in k.lower()]
        decoder_keys = [k for k in state_dict.keys() if 'decoder' in k.lower()]
        
        print(f"   Encoder layers: {len(encoder_keys)}")
        print(f"   Decoder layers: {len(decoder_keys)}")
        
        # Look for latent dimension
        for key, weight in state_dict.items():
            if 'mean' in key or 'mu' in key:
                if len(weight.shape) == 1:  # Bias term
                    latent_dim = weight.shape[0]
                    print(f"   Inferred latent dimension: {latent_dim}")
                    break
        
        # Look for input/output dimensions
        for key, weight in state_dict.items():
            if weight.dim() == 2:  # Linear layer
                print(f"   Layer {key}: {weight.shape}")
                if 'bodyprior_enc' in key:
                    print(f"     → Encoder input dim: {weight.shape[1]}")
                elif 'bodyprior_dec' in key and 'weight' in key:
                    print(f"     → Decoder output dim: {weight.shape[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Architecture inference failed: {e}")
        return False

def main():
    """Run direct VPoser tests"""
    print("🚀 Direct VPoser Model Tests")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    tests = [
        test_model_architecture_inference,
        test_direct_vposer_load,
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)