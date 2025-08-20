"""
Test loss function implementations
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.loss import CoordLoss, ParamLoss


class TestCoordLoss:
    """Test coordinate loss function"""
    
    def test_coord_loss_basic(self):
        """Test basic coordinate loss functionality"""
        loss_fn = CoordLoss()
        batch_size, num_joints = 4, 10
        
        # Perfect predictions (zero loss)
        coords_pred = torch.randn(batch_size, num_joints, 3)
        coords_gt = coords_pred.clone()
        valid = torch.ones(batch_size, num_joints, 3)
        
        loss = loss_fn(coords_pred, coords_gt, valid)
        
        assert loss.shape == (batch_size, num_joints, 3)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
    
    def test_coord_loss_with_error(self):
        """Test coordinate loss with prediction errors"""
        loss_fn = CoordLoss()
        batch_size, num_joints = 2, 5
        
        # Create predictions with known error
        coords_gt = torch.zeros(batch_size, num_joints, 3)
        coords_pred = torch.ones(batch_size, num_joints, 3)  # Error of 1.0 per coordinate
        valid = torch.ones(batch_size, num_joints, 3)
        
        loss = loss_fn(coords_pred, coords_gt, valid)
        
        # Loss should be absolute difference = 1.0
        expected_loss = torch.ones(batch_size, num_joints, 3)
        assert torch.allclose(loss, expected_loss)
    
    def test_coord_loss_masking(self):
        """Test coordinate loss with validity masking"""
        loss_fn = CoordLoss()
        batch_size, num_joints = 3, 4
        
        coords_pred = torch.ones(batch_size, num_joints, 3)
        coords_gt = torch.zeros(batch_size, num_joints, 3)
        
        # Create validity mask - only first joint is valid
        valid = torch.zeros(batch_size, num_joints, 3)
        valid[:, 0, :] = 1.0
        
        loss = loss_fn(coords_pred, coords_gt, valid)
        
        # Only first joint should have non-zero loss
        assert torch.allclose(loss[:, 1:, :], torch.zeros(batch_size, num_joints-1, 3))
        assert torch.allclose(loss[:, 0, :], torch.ones(batch_size, 3))
    
    def test_coord_loss_3d_masking(self):
        """Test coordinate loss with 3D validity masking"""
        loss_fn = CoordLoss()
        batch_size, num_joints = 2, 6
        
        coords_pred = torch.ones(batch_size, num_joints, 3)
        coords_gt = torch.zeros(batch_size, num_joints, 3)
        valid = torch.ones(batch_size, num_joints, 3)
        
        # Only first batch element has 3D ground truth
        is_3D = torch.tensor([1.0, 0.0])
        
        loss = loss_fn(coords_pred, coords_gt, valid, is_3D)
        
        # Check Z-dimension masking
        # First batch: full 3D loss
        assert torch.allclose(loss[0, :, :], torch.ones(num_joints, 3))
        # Second batch: only XY loss, Z should be zero
        assert torch.allclose(loss[1, :, :2], torch.ones(num_joints, 2))
        assert torch.allclose(loss[1, :, 2], torch.zeros(num_joints))
    
    def test_coord_loss_gradient_flow(self):
        """Test gradient flow through coordinate loss"""
        loss_fn = CoordLoss()
        batch_size, num_joints = 2, 3
        
        coords_pred = torch.randn(batch_size, num_joints, 3, requires_grad=True)
        coords_gt = torch.randn(batch_size, num_joints, 3)
        valid = torch.ones(batch_size, num_joints, 3)
        
        loss = loss_fn(coords_pred, coords_gt, valid)
        total_loss = loss.sum()
        
        total_loss.backward()
        
        # Check gradients exist and are reasonable
        assert coords_pred.grad is not None
        assert not torch.allclose(coords_pred.grad, torch.zeros_like(coords_pred.grad))


class TestParamLoss:
    """Test parameter loss function"""
    
    def test_param_loss_basic(self):
        """Test basic parameter loss functionality"""
        loss_fn = ParamLoss()
        batch_size, param_dim = 5, 10
        
        # Perfect predictions (zero loss)
        params_pred = torch.randn(batch_size, param_dim)
        params_gt = params_pred.clone()
        valid = torch.ones(batch_size, param_dim)
        
        loss = loss_fn(params_pred, params_gt, valid)
        
        assert loss.shape == (batch_size, param_dim)
        assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)
    
    def test_param_loss_with_error(self):
        """Test parameter loss with prediction errors"""
        loss_fn = ParamLoss()
        batch_size, param_dim = 3, 5
        
        # Create predictions with known error
        params_gt = torch.zeros(batch_size, param_dim)
        params_pred = torch.ones(batch_size, param_dim) * 2.0  # Error of 2.0
        valid = torch.ones(batch_size, param_dim)
        
        loss = loss_fn(params_pred, params_gt, valid)
        
        # Loss should be absolute difference = 2.0
        expected_loss = torch.ones(batch_size, param_dim) * 2.0
        assert torch.allclose(loss, expected_loss)
    
    def test_param_loss_masking(self):
        """Test parameter loss with validity masking"""
        loss_fn = ParamLoss()
        batch_size, param_dim = 4, 8
        
        params_pred = torch.ones(batch_size, param_dim)
        params_gt = torch.zeros(batch_size, param_dim)
        
        # Create validity mask - only even indices are valid
        valid = torch.zeros(batch_size, param_dim)
        valid[:, ::2] = 1.0
        
        loss = loss_fn(params_pred, params_gt, valid)
        
        # Only even indices should have non-zero loss
        assert torch.allclose(loss[:, 1::2], torch.zeros(batch_size, param_dim//2))
        assert torch.allclose(loss[:, ::2], torch.ones(batch_size, param_dim//2))
    
    def test_param_loss_gradient_flow(self):
        """Test gradient flow through parameter loss"""
        loss_fn = ParamLoss()
        batch_size, param_dim = 2, 6
        
        params_pred = torch.randn(batch_size, param_dim, requires_grad=True)
        params_gt = torch.randn(batch_size, param_dim)
        valid = torch.ones(batch_size, param_dim)
        
        loss = loss_fn(params_pred, params_gt, valid)
        total_loss = loss.sum()
        
        total_loss.backward()
        
        # Check gradients exist and are reasonable
        assert params_pred.grad is not None
        assert not torch.allclose(params_pred.grad, torch.zeros_like(params_pred.grad))


class TestLossIntegration:
    """Test loss functions in integrated scenarios"""
    
    def test_loss_scaling(self):
        """Test loss function behavior with different scales"""
        coord_loss_fn = CoordLoss()
        param_loss_fn = ParamLoss()
        
        batch_size = 3
        
        # Test with different scales
        scales = [0.1, 1.0, 10.0]
        
        for scale in scales:
            # Coordinate loss
            coords_pred = torch.zeros(batch_size, 5, 3)
            coords_gt = torch.ones(batch_size, 5, 3) * scale
            valid = torch.ones(batch_size, 5, 3)
            
            coord_loss = coord_loss_fn(coords_pred, coords_gt, valid)
            assert torch.allclose(coord_loss, torch.ones_like(coord_loss) * scale)
            
            # Parameter loss
            params_pred = torch.zeros(batch_size, 10)
            params_gt = torch.ones(batch_size, 10) * scale
            valid = torch.ones(batch_size, 10)
            
            param_loss = param_loss_fn(params_pred, params_gt, valid)
            assert torch.allclose(param_loss, torch.ones_like(param_loss) * scale)
    
    def test_loss_numerical_stability(self):
        """Test loss functions with extreme values"""
        coord_loss_fn = CoordLoss()
        param_loss_fn = ParamLoss()
        
        batch_size = 2
        
        # Test with very large values
        large_val = 1e6
        coords_pred = torch.ones(batch_size, 3, 3) * large_val
        coords_gt = torch.zeros(batch_size, 3, 3)
        valid = torch.ones(batch_size, 3, 3)
        
        coord_loss = coord_loss_fn(coords_pred, coords_gt, valid)
        assert torch.all(torch.isfinite(coord_loss))
        
        # Test with very small values
        small_val = 1e-8
        params_pred = torch.ones(batch_size, 5) * small_val
        params_gt = torch.zeros(batch_size, 5)
        valid = torch.ones(batch_size, 5)
        
        param_loss = param_loss_fn(params_pred, params_gt, valid)
        assert torch.all(torch.isfinite(param_loss))
        assert torch.allclose(param_loss, torch.ones_like(param_loss) * small_val)
    
    def test_batch_size_consistency(self):
        """Test loss functions with different batch sizes"""
        coord_loss_fn = CoordLoss()
        param_loss_fn = ParamLoss()
        
        for batch_size in [1, 2, 8, 16]:
            # Coordinate loss
            coords_pred = torch.randn(batch_size, 4, 3)
            coords_gt = torch.randn(batch_size, 4, 3)
            valid = torch.ones(batch_size, 4, 3)
            
            coord_loss = coord_loss_fn(coords_pred, coords_gt, valid)
            assert coord_loss.shape == (batch_size, 4, 3)
            
            # Parameter loss
            params_pred = torch.randn(batch_size, 6)
            params_gt = torch.randn(batch_size, 6)
            valid = torch.ones(batch_size, 6)
            
            param_loss = param_loss_fn(params_pred, params_gt, valid)
            assert param_loss.shape == (batch_size, 6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])