#!/usr/bin/env python3
"""
Pre-flight check script for SMPLest-X inference on Mac
Validates all dependencies, model files, and system compatibility
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requires Python 3.8+)")
        return False

def check_pytorch():
    """Check PyTorch installation and device support"""
    print("\n🔥 Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        
        # Check device support
        if torch.backends.mps.is_available():
            print("   ✅ MPS (Apple Silicon) support available")
            device = "mps"
        elif torch.cuda.is_available():
            print("   ✅ CUDA support available")
            device = "cuda"
        else:
            print("   ⚠️  Using CPU (slower performance expected)")
            device = "cpu"
            
        # Test device functionality
        test_tensor = torch.tensor([1.0, 2.0]).to(device)
        print(f"   ✅ Device '{device}' working correctly")
        return True, device
        
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False, None
    except Exception as e:
        print(f"   ❌ PyTorch device error: {e}")
        return False, None

def check_dependencies():
    """Check required Python packages"""
    print("\n📦 Checking dependencies...")
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'ultralytics': 'ultralytics',
        'smplx': 'smplx',
        'trimesh': 'trimesh',
        'pyrender': 'pyrender',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm',
        'einops': 'einops',
        'huggingface_hub': 'huggingface_hub'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {pip_name}: {version}")
        except ImportError:
            print(f"   ❌ {pip_name}: not installed")
            missing.append(pip_name)
    
    if missing:
        print(f"\n   💡 Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

def check_ffmpeg():
    """Check FFmpeg installation for video processing"""
    print("\n🎬 Checking FFmpeg...")
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ {version_line}")
            return True
        else:
            print("   ❌ FFmpeg not working properly")
            return False
    except FileNotFoundError:
        print("   ❌ FFmpeg not installed")
        print("   💡 Install with: brew install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("   ❌ FFmpeg check timed out")
        return False

def check_model_files():
    """Check if model files are present"""
    print("\n🤖 Checking model files...")
    
    model_files = {
        'pretrained_models/smplest_x_h/smplest_x_h.pth.tar': 'Main model weights',
        'pretrained_models/smplest_x_h/config_base.py': 'Configuration file'
    }
    
    all_present = True
    for file_path, description in model_files.items():
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {description}: {size_mb:.1f} MB")
        else:
            print(f"   ❌ {description}: missing")
            print(f"       Expected: {file_path}")
            all_present = False
    
    if not all_present:
        print("   💡 Run: python download_weights.py")
    
    return all_present

def check_demo_setup():
    """Check demo directory structure"""
    print("\n📁 Checking demo setup...")
    
    demo_dir = Path('demo')
    if not demo_dir.exists():
        print("   ⚠️  Demo directory missing, creating...")
        demo_dir.mkdir()
        print("   ✅ Demo directory created")
    else:
        print("   ✅ Demo directory exists")
    
    # Check for any existing demo files
    demo_files = list(demo_dir.glob('*.mp4')) + list(demo_dir.glob('*.mov')) + list(demo_dir.glob('*.avi'))
    if demo_files:
        print(f"   📹 Found {len(demo_files)} video file(s):")
        for file in demo_files[:3]:  # Show first 3
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"       • {file.name} ({size_mb:.1f} MB)")
        if len(demo_files) > 3:
            print(f"       ... and {len(demo_files) - 3} more")
    else:
        print("   💡 No demo videos found. Place test videos in demo/ directory")
    
    return True

def check_human_models():
    """Check SMPL/SMPL-X model files"""
    print("\n👤 Checking human models...")
    
    model_dir = Path('human_models/human_model_files')
    if not model_dir.exists():
        print("   ⚠️  Human models directory missing")
        print("   💡 Download SMPL/SMPL-X models from official websites")
        return False
    
    smpl_files = list(model_dir.glob('*.pkl'))
    smplx_files = list((model_dir / 'smplx').glob('*.npz')) if (model_dir / 'smplx').exists() else []
    
    if smpl_files or smplx_files:
        if smpl_files:
            print(f"   ✅ Found {len(smpl_files)} SMPL model file(s)")
        if smplx_files:
            print(f"   ✅ Found {len(smplx_files)} SMPL-X model file(s)")
        return True
    else:
        print("   ⚠️  No SMPL/SMPL-X model files found")
        print("   💡 Download from https://smpl.is.tue.mpg.de/ and https://smpl-x.is.tue.mpg.de/")
        return False

def check_mac_specific():
    """Mac-specific checks"""
    print("\n🍎 Mac-specific checks...")
    
    # Check macOS version
    try:
        result = subprocess.run(['sw_vers', '-productVersion'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✅ macOS {version}")
        else:
            print("   ⚠️  Could not determine macOS version")
    except:
        print("   ⚠️  Could not check macOS version")
    
    # Check for common Mac issues
    if 'PYTORCH_ENABLE_MPS_FALLBACK' not in os.environ:
        print("   💡 Consider setting: export PYTORCH_ENABLE_MPS_FALLBACK=1")
    
    return True

def main():
    """Run all pre-flight checks"""
    print("🚀 SMPLest-X Pre-flight Check for Mac")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & Devices", check_pytorch), 
        ("Dependencies", check_dependencies),
        ("FFmpeg", check_ffmpeg),
        ("Model Files", check_model_files),
        ("Demo Setup", check_demo_setup),
        ("Human Models", check_human_models),
        ("Mac Specific", check_mac_specific)
    ]
    
    results = {}
    device = None
    
    for name, check_func in checks:
        try:
            if name == "PyTorch & Devices":
                result, device = check_func()
                results[name] = result
            else:
                results[name] = check_func()
        except Exception as e:
            print(f"   ❌ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Summary")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")
    
    print(f"\n🎯 Overall: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready for inference.")
        if device:
            print(f"💡 Recommended device: {device}")
        return True
    else:
        print(f"\n⚠️  {total - passed} issue(s) need attention before running inference.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)