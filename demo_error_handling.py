"""
Demonstration of enhanced checkpoint loading with error handling.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.SMPLest_X import get_model
from configs import config_smplest_x_h as cfg_module
from utils.checkpoint_utils import (
    validate_checkpoint_file, 
    load_checkpoint_safely,
    get_checkpoint_info
)


class ConfigObj:
    """Convert config dict to object with proper attribute access"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObj(value))
            else:
                setattr(self, key, value)


def demo_error_handling():
    """Demonstrate various error handling scenarios"""
    
    print("🔍 SMPLest-X Checkpoint Loading Error Handling Demo")
    print("=" * 60)
    
    # Test 1: Valid checkpoint loading
    print("\n📋 Test 1: Valid checkpoint loading")
    print("-" * 30)
    
    valid_checkpoint = Path('./pretrained_models/smplest_x_h/smplest_x_h.pth.tar')
    
    if validate_checkpoint_file(valid_checkpoint):
        print(f"✅ Checkpoint validation passed: {valid_checkpoint}")
        info = get_checkpoint_info(valid_checkpoint)
        print(f"   Size: {info['size_mb']} MB")
        print(f"   Epoch: {info['epoch']}")
        print(f"   Parameters: {info['parameter_count']:,}")
    else:
        print(f"❌ Checkpoint validation failed: {valid_checkpoint}")
    
    # Test 2: Missing checkpoint handling
    print("\n📋 Test 2: Missing checkpoint handling")
    print("-" * 30)
    
    missing_checkpoint = Path('./non_existent_checkpoint.pth')
    
    if validate_checkpoint_file(missing_checkpoint):
        print(f"✅ Checkpoint found: {missing_checkpoint}")
    else:
        print(f"⚠️  Checkpoint not found: {missing_checkpoint} (handled gracefully)")
    
    checkpoint_data = load_checkpoint_safely(missing_checkpoint)
    if checkpoint_data is None:
        print("✅ Missing checkpoint handled gracefully - returned None")
    
    # Test 3: Model creation with valid checkpoint
    print("\n📋 Test 3: Model creation with checkpoint loading")
    print("-" * 30)
    
    try:
        cfg = ConfigObj(cfg_module.config)
        cfg.model.pretrained_model_path = str(valid_checkpoint)
        
        print("Creating model in test mode...")
        model = get_model(cfg, mode='test')
        
        print("✅ Model creation successful!")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
    
    # Test 4: Model creation with missing checkpoint  
    print("\n📋 Test 4: Model creation with missing checkpoint")
    print("-" * 30)
    
    try:
        cfg = ConfigObj(cfg_module.config)
        cfg.model.pretrained_model_path = str(missing_checkpoint)
        
        print("Attempting to create model with missing checkpoint...")
        model = get_model(cfg, mode='test')
        
        print("⚠️  This shouldn't happen - test mode requires checkpoint")
        
    except FileNotFoundError as e:
        print("✅ Missing checkpoint correctly detected and handled:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test 5: Model creation in train mode (should work without checkpoint)
    print("\n📋 Test 5: Train mode without checkpoint")
    print("-" * 30)
    
    try:
        cfg = ConfigObj(cfg_module.config)
        # Remove the checkpoint path for train mode
        if hasattr(cfg.model, 'pretrained_model_path'):
            delattr(cfg.model, 'pretrained_model_path')
        
        print("Creating model in train mode without checkpoint...")
        model = get_model(cfg, mode='train')
        
        print("✅ Train mode model creation successful!")
        print("   (Train mode doesn't require main checkpoint - uses random initialization)")
        
    except Exception as e:
        print(f"❌ Train mode creation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Error handling demonstration completed!")
    print("\nKey features demonstrated:")
    print("• Checkpoint validation before loading")
    print("• Graceful handling of missing files")
    print("• Informative error messages")
    print("• Fallback to random initialization when appropriate")
    print("• Comprehensive logging and status reporting")


if __name__ == "__main__":
    demo_error_handling()