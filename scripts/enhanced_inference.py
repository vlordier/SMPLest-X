#!/usr/bin/env python3
"""
Enhanced inference script with Mac-specific error handling and robustness checks
"""

import os
import sys
import argparse
import traceback
from pathlib import Path

def setup_mac_environment():
    """Configure Mac-specific environment variables"""
    if not os.getenv('PYTORCH_ENABLE_MPS_FALLBACK'):
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        print("🍎 Set PYTORCH_ENABLE_MPS_FALLBACK=1 for Mac compatibility")
    
    # Prevent OpenMP issues on Mac
    if not os.getenv('OMP_NUM_THREADS'):
        os.environ['OMP_NUM_THREADS'] = '1'
    
    # Disable MPS graph optimization for better compatibility
    if not os.getenv('PYTORCH_MPS_HIGH_WATERMARK_RATIO'):
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def validate_prerequisites():
    """Validate all prerequisites before inference"""
    print("🔍 Validating prerequisites...")
    
    errors = []
    warnings = []
    
    # Check model files
    model_files = [
        'pretrained_models/smplest_x_h/smplest_x_h.pth.tar',
        'pretrained_models/smplest_x_h/config_base.py'
    ]
    
    for file_path in model_files:
        if not os.path.exists(file_path):
            errors.append(f"Missing model file: {file_path}")
    
    # Check demo directory and input
    if not os.path.exists('demo'):
        errors.append("Missing demo directory")
    
    # Check PyTorch and device
    try:
        import torch
        device = torch.device('mps' if torch.backends.mps.is_available() else 
                             'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   ✅ Using device: {device}")
        
        # Test device functionality
        test_tensor = torch.tensor([1.0]).to(device)
        _ = test_tensor * 2  # Simple operation
        print(f"   ✅ Device {device} working correctly")
        
    except Exception as e:
        errors.append(f"PyTorch device error: {e}")
    
    # Check dependencies
    required_modules = ['cv2', 'numpy', 'ultralytics', 'smplx', 'trimesh']
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            errors.append(f"Missing dependency: {module}")
    
    # Check FFmpeg
    import shutil
    if not shutil.which('ffmpeg'):
        warnings.append("FFmpeg not found - video processing may fail")
    
    if errors:
        print("❌ Prerequisites check failed:")
        for error in errors:
            print(f"   • {error}")
        if warnings:
            print("⚠️  Warnings:")
            for warning in warnings:
                print(f"   • {warning}")
        return False
    
    if warnings:
        print("⚠️  Warnings (inference may still work):")
        for warning in warnings:
            print(f"   • {warning}")
    
    print("✅ All prerequisites validated")
    return True

def run_inference_with_error_handling(model_dir, file_name, fps):
    """Run inference with comprehensive error handling"""
    print(f"\n🚀 Starting inference: {file_name} (model: {model_dir}, fps: {fps})")
    
    try:
        # Import inference modules with error handling
        sys.path.insert(0, os.getcwd())
        
        print("📦 Loading SMPLest-X modules...")
        from main.inference import main as inference_main
        
        # Backup original sys.argv
        original_argv = sys.argv.copy()
        
        # Set up arguments for inference
        sys.argv = [
            'inference.py',
            '--num_gpus', '1',
            '--file_name', file_name,
            '--ckpt_name', model_dir,
            '--end', str(get_frame_count(file_name))
        ]
        
        print("🎬 Running inference...")
        
        # Run inference with timeout protection
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Inference timed out after 10 minutes")
        
        # Set 10 minute timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes
        
        try:
            inference_main()
            print("✅ Inference completed successfully")
            return True
        finally:
            signal.alarm(0)  # Cancel timeout
            sys.argv = original_argv  # Restore original argv
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Check if all model files and input data exist")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "MPS" in error_msg:
            print(f"❌ MPS error: {e}")
            print("💡 Try setting PYTORCH_ENABLE_MPS_FALLBACK=1")
            print("💡 Or fallback to CPU with: export CUDA_VISIBLE_DEVICES=''")
        elif "CUDA" in error_msg:
            print(f"❌ CUDA error: {e}")
            print("💡 CUDA not available on Mac, using MPS/CPU")
        else:
            print(f"❌ Runtime error: {e}")
        return False
    except TimeoutError as e:
        print(f"❌ {e}")
        print("💡 Try with a shorter video or simpler settings")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

def get_frame_count(file_name):
    """Get frame count from input frames directory"""
    input_dir = Path(f'demo/input_frames/{file_name}')
    if input_dir.exists():
        frame_files = list(input_dir.glob('*.jpg'))
        return len(frame_files)
    return 1

def main():
    parser = argparse.ArgumentParser(description='Enhanced SMPLest-X inference for Mac')
    parser.add_argument('model_dir', help='Model directory name (e.g., smplest_x_h)')
    parser.add_argument('file_name', help='Input file name (e.g., test_person.mp4)')
    parser.add_argument('fps', type=int, help='Frames per second')
    parser.add_argument('--skip-checks', action='store_true', help='Skip prerequisite validation')
    
    args = parser.parse_args()
    
    print("🤖 Enhanced SMPLest-X Inference for Mac")
    print("=" * 50)
    
    # Setup Mac environment
    setup_mac_environment()
    
    # Validate prerequisites
    if not args.skip_checks:
        if not validate_prerequisites():
            print("\n❌ Prerequisites validation failed. Fix issues and try again.")
            print("💡 Use --skip-checks to bypass validation (not recommended)")
            return False
    
    # Extract file name without extension
    file_stem = Path(args.file_name).stem
    
    # Run inference with error handling
    success = run_inference_with_error_handling(args.model_dir, file_stem, args.fps)
    
    if success:
        print("\n🎉 Inference completed successfully!")
        print(f"📁 Check output in: demo/result_{file_stem}.mp4")
        return True
    else:
        print("\n💥 Inference failed. Check errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)