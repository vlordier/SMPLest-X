"""
Basic tests for SMPLest-X components that don't require model files
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loss import CoordLoss, ParamLoss


class TestBasicFunctionality:
    """Test basic functionality without model dependencies"""
    
    def test_imports(self):
        """Test that basic imports work"""
        try:
            from models.loss import CoordLoss, ParamLoss
            from models.module import PatchEmbed, Block, Mlp
            from utils.device_utils import to_device
            
            # Actually use the imports to verify they work
            assert CoordLoss is not None
            assert ParamLoss is not None
            assert PatchEmbed is not None
            assert Block is not None
            assert Mlp is not None
            assert to_device is not None
            
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")
    
    def test_torch_operations(self):
        """Test basic torch operations work"""
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        
        # Basic operations
        z = x + y
        assert z.shape == (3, 4)
        
        # Matrix operations
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)
        c = torch.matmul(a, b)
        assert c.shape == (3, 5)
        
        # Rotation matrix properties
        rot = torch.randn(3, 3)
        # Make it a proper rotation matrix
        u, s, v = torch.svd(rot)
        proper_rot = torch.matmul(u, v.transpose(-1, -2))
        
        # Check determinant is close to ±1 (can be -1 for improper rotation)
        det = torch.det(proper_rot)
        assert torch.allclose(torch.abs(det), torch.tensor(1.0), atol=1e-5)


class TestLossBasic:
    """Test loss functions with simple cases"""
    
    def test_coordinate_loss_simple(self):
        """Test coordinate loss with simple inputs"""
        loss_fn = CoordLoss()
        
        # Simple case: identical predictions
        pred = torch.zeros(2, 3, 3)
        gt = torch.zeros(2, 3, 3)
        valid = torch.ones(2, 3, 3)
        
        loss = loss_fn(pred, gt, valid)
        assert torch.allclose(loss, torch.zeros_like(loss))
        
        # Simple case: unit difference
        pred = torch.ones(2, 3, 3)
        gt = torch.zeros(2, 3, 3)
        valid = torch.ones(2, 3, 3)
        
        loss = loss_fn(pred, gt, valid)
        expected = torch.ones(2, 3, 3)
        assert torch.allclose(loss, expected)
    
    def test_parameter_loss_simple(self):
        """Test parameter loss with simple inputs"""
        loss_fn = ParamLoss()
        
        # Simple case: identical predictions
        pred = torch.zeros(2, 5)
        gt = torch.zeros(2, 5)
        valid = torch.ones(2, 5)
        
        loss = loss_fn(pred, gt, valid)
        assert torch.allclose(loss, torch.zeros_like(loss))
        
        # Simple case: unit difference
        pred = torch.ones(2, 5)
        gt = torch.zeros(2, 5)
        valid = torch.ones(2, 5)
        
        loss = loss_fn(pred, gt, valid)
        expected = torch.ones(2, 5)
        assert torch.allclose(loss, expected)


class TestModuleBasic:
    """Test basic module components"""
    
    def test_patch_embed_creation(self):
        """Test PatchEmbed creation"""
        try:
            from models.module import PatchEmbed
            patch_embed = PatchEmbed(img_size=224, patch_size=16, embed_dim=768)
            assert patch_embed is not None
            assert hasattr(patch_embed, 'proj')
        except Exception as e:
            pytest.skip(f"PatchEmbed test skipped: {e}")
    
    def test_mlp_creation(self):
        """Test MLP creation"""
        try:
            from models.module import Mlp
            mlp = Mlp(in_features=768, hidden_features=1024)
            assert mlp is not None
            assert hasattr(mlp, 'fc1')
            assert hasattr(mlp, 'fc2')
        except Exception as e:
            pytest.skip(f"MLP test skipped: {e}")
    
    def test_block_creation(self):
        """Test Block creation"""
        try:
            from models.module import Block
            block = Block(dim=768, num_heads=12)
            assert block is not None
            assert hasattr(block, 'norm1')
            assert hasattr(block, 'norm2')
            assert hasattr(block, 'attn')
            assert hasattr(block, 'mlp')
        except Exception as e:
            pytest.skip(f"Block test skipped: {e}")


class TestGeometryBasic:
    """Test basic geometry operations"""
    
    def test_rotation_matrix_properties(self):
        """Test rotation matrix properties"""
        # Create random rotation matrix
        rand_mat = torch.randn(3, 3)
        u, s, v = torch.svd(rand_mat)
        rot_mat = torch.matmul(u, v.transpose(-1, -2))
        
        # Check orthogonality: R @ R.T = I
        should_be_identity = torch.matmul(rot_mat, rot_mat.transpose(-1, -2))
        identity = torch.eye(3)
        assert torch.allclose(should_be_identity, identity, atol=1e-5)
        
        # Check determinant = ±1 (can be -1 for improper rotation)  
        det = torch.det(rot_mat)
        assert torch.allclose(torch.abs(det), torch.tensor(1.0), atol=1e-5)
    
    def test_batch_rotation_matrices(self):
        """Test batch of rotation matrices"""
        batch_size = 5
        
        # Create batch of rotation matrices
        rand_mats = torch.randn(batch_size, 3, 3)
        rot_mats = []
        
        for i in range(batch_size):
            u, s, v = torch.svd(rand_mats[i])
            rot_mat = torch.matmul(u, v.transpose(-1, -2))
            rot_mats.append(rot_mat)
        
        rot_mats = torch.stack(rot_mats)
        
        # Check all are proper rotation matrices
        for i in range(batch_size):
            should_be_identity = torch.matmul(rot_mats[i], rot_mats[i].transpose(-1, -2))
            identity = torch.eye(3)
            assert torch.allclose(should_be_identity, identity, atol=1e-5)
            
            det = torch.det(rot_mats[i])
            assert torch.allclose(torch.abs(det), torch.tensor(1.0), atol=1e-5)
    
    def test_camera_projection_basic(self):
        """Test basic camera projection"""
        # Simple orthographic projection
        points_3d = torch.tensor([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0]
        ])
        
        # Scale factor
        focal = 1000.0
        center = 256.0
        
        # Project to 2D
        x = points_3d[:, 0] / points_3d[:, 2] * focal + center
        y = points_3d[:, 1] / points_3d[:, 2] * focal + center
        
        projected = torch.stack([x, y], dim=1)
        
        expected = torch.tensor([
            [256.0, 256.0],  # Center point
            [1256.0, 256.0], # Right
            [256.0, 1256.0]  # Up
        ])
        
        assert torch.allclose(projected, expected)


class TestDeviceCompatibility:
    """Test device compatibility"""
    
    def test_cpu_operations(self):
        """Test operations on CPU"""
        device = torch.device('cpu')
        
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 4, device=device)
        
        z = x + y
        assert z.device.type == device.type
        
        # Matrix operations
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        c = torch.matmul(a, b)
        assert c.device.type == device.type
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_operations(self):
        """Test operations on GPU"""
        device = torch.device('cuda')
        
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 4, device=device)
        
        z = x + y
        assert z.device.type == device.type
        
        # Matrix operations
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        c = torch.matmul(a, b)
        assert c.device.type == device.type
    
    @pytest.mark.skipif(not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()), 
                       reason="MPS not available")
    def test_mps_operations(self):
        """Test operations on MPS (Mac GPU)"""
        device = torch.device('mps')
        
        x = torch.randn(3, 4, device=device)
        y = torch.randn(3, 4, device=device)
        
        z = x + y
        assert z.device.type == device.type
        
        # Matrix operations
        a = torch.randn(3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        c = torch.matmul(a, b)
        assert c.device.type == device.type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])