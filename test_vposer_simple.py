"""
Simple VPoser availability test
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_vposer_imports():
    """Test if VPoser can be imported"""
    try:
        from human_body_prior.tools.model_loader import load_model
        from human_body_prior.models.vposer_model import VPoser
        print("✅ VPoser imports successful")
        return True
    except ImportError as e:
        print(f"❌ VPoser import failed: {e}")
        return False

def test_pytorch3d_imports():
    """Test if PyTorch3D can be imported"""
    try:
        from pytorch3d.structures import Meshes
        print("✅ PyTorch3D imports successful")
        return True
    except ImportError as e:
        print(f"❌ PyTorch3D import failed: {e}")
        return False

def test_smplx_availability():
    """Test SMPL-X model availability"""
    try:
        from human_models.human_models import SMPLX
        smplx = SMPLX.get_instance()
        print(f"✅ SMPL-X available: {smplx.vertex_num} vertices")
        return True
    except Exception as e:
        print(f"❌ SMPL-X failed: {e}")
        return False

def test_vposer_model_directory():
    """Check VPoser model directory"""
    vposer_dir = Path('./data/vposer_v1_0/snapshots')
    print(f"📁 VPoser directory: {vposer_dir.absolute()}")
    print(f"📁 Directory exists: {vposer_dir.exists()}")
    
    if vposer_dir.exists():
        files = list(vposer_dir.glob('*'))
        print(f"📁 Files in directory: {len(files)}")
        for f in files[:5]:  # Show first 5 files
            print(f"   - {f.name}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more")
    
    return vposer_dir.exists()

def main():
    """Run simple availability tests"""
    print("🔍 Testing VPoser Integration Availability")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print()
    
    tests = [
        ("VPoser Imports", test_vposer_imports),
        ("PyTorch3D Imports", test_pytorch3d_imports),
        ("SMPL-X Availability", test_smplx_availability),
        ("VPoser Model Directory", test_vposer_model_directory),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"🧪 {test_name}:")
        if test_func():
            passed += 1
        print()
    
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All components available!")
    else:
        print("⚠️  Some components missing - check setup")
        
        print("\n💡 Next Steps:")
        if passed < 2:
            print("   - VPoser model needs to be downloaded from SMPL-X website")
            print("   - Register at https://smpl-x.is.tue.mpg.de/")
            print("   - Download 'VPoser: Variational Human Pose Prior'")
            print("   - Extract to ./data/vposer_v1_0/snapshots/")
        
        if "PyTorch3D" in [test[0] for test in tests if not passed]:
            print("   - PyTorch3D installation may require specific PyTorch version")
            print("   - Can proceed with PyVista fallback rendering")

if __name__ == "__main__":
    main()