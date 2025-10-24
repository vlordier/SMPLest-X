"""
Test checkpoint integration with the actual codebase.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.SMPLest_X import get_model
from configs import config_smplest_x_h as cfg_module
from utils.checkpoint_utils import get_checkpoint_info


class ConfigObj:
    """Convert config dict to object with proper attribute access"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObj(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self):
        """Convert back to dictionary for ** unpacking"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigObj):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def test_checkpoint_integration():
    """Test checkpoint loading and integration"""
    
    print("🔍 Testing SMPLest-X Checkpoint Integration")
    print("=" * 50)
    
    # Convert config to object
    cfg = ConfigObj(cfg_module.config)
    
    # Override the checkpoint path to use the actual available checkpoint
    cfg.model.pretrained_model_path = './pretrained_models/smplest_x_h/smplest_x_h.pth.tar'
    
    # Check checkpoint file
    checkpoint_path = Path(cfg.model.pretrained_model_path)
    print(f"📁 Checkpoint path: {checkpoint_path}")
    print("📊 Checkpoint info:")
    
    info = get_checkpoint_info(checkpoint_path)
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "=" * 50)
    
    # Test model creation in test mode
    print("🧪 Testing TEST mode (with checkpoint loading)...")
    try:
        model = get_model(cfg, mode='test')
        print("✅ Model created successfully!")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("📈 Model statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / (1024**3):.2f} GB (fp32)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    
    # Test model creation in train mode
    print("🏋️ Testing TRAIN mode (with encoder checkpoint if available)...")
    try:
        model = get_model(cfg, mode='train')
        print("✅ Model created successfully!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ Checkpoint integration test completed!")


if __name__ == "__main__":
    test_checkpoint_integration()