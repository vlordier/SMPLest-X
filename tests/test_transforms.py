"""
Test transformation utilities and coordinate conversions
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.transforms import rot6d_to_axis_angle, batch_rodrigues


class TestRotationTransforms:
    """Test rotation transformations"""
    
    def test_rot6d_to_axis_angle_shape(self):
        """Test 6D rotation to axis-angle conversion shapes"""
        batch_size = 5
        
        # Test single rotation (needs batch dimension)
        rot6d_single = torch.randn(1, 6)
        axis_angle = rot6d_to_axis_angle(rot6d_single)
        assert axis_angle.shape == (1, 3)
        
        # Test batch rotations
        rot6d_batch = torch.randn(batch_size, 6)
        axis_angle_batch = rot6d_to_axis_angle(rot6d_batch)
        assert axis_angle_batch.shape == (batch_size, 3)
    
    def test_rot6d_identity(self):
        """Test identity rotation conversion"""
        # Create proper 6D representation from identity matrix
        # Identity matrix first two columns: [1,0,0] and [0,1,0]
        identity_rot = torch.eye(3)
        identity_6d = identity_rot[:, :2].reshape(1, -1)  # Shape: [1, 6]
        
        axis_angle = rot6d_to_axis_angle(identity_6d)
        
        # Should be close to zero rotation
        assert torch.allclose(axis_angle, torch.zeros(1, 3), atol=1e-5)
    
    def test_batch_rodrigues_shape(self):
        """Test batch Rodrigues formula shapes"""
        batch_size = 3
        
        # Test single rotation (needs batch dimension)
        axis_angle_single = torch.randn(1, 3) * 0.1
        rot_mat = batch_rodrigues(axis_angle_single)
        assert rot_mat.shape == (1, 3, 3)
        
        # Test batch rotations
        axis_angle_batch = torch.randn(batch_size, 3) * 0.1
        rot_mat_batch = batch_rodrigues(axis_angle_batch)
        assert rot_mat_batch.shape == (batch_size, 3, 3)
    
    def test_rodrigues_identity(self):
        """Test identity rotation with Rodrigues formula"""
        zero_rotation = torch.zeros(1, 3)  # Add batch dimension
        rot_mat = batch_rodrigues(zero_rotation)
        identity = torch.eye(3).unsqueeze(0)  # Add batch dimension
        
        assert torch.allclose(rot_mat, identity, atol=1e-6)
    
    def test_rodrigues_orthogonality(self):
        """Test that Rodrigues formula produces orthogonal matrices"""
        batch_size = 10
        axis_angles = torch.randn(batch_size, 3) * 0.5
        
        rot_mats = batch_rodrigues(axis_angles)
        
        # Check orthogonality: R @ R.T = I
        identity_batch = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        should_be_identity = torch.matmul(rot_mats, rot_mats.transpose(-1, -2))
        
        assert torch.allclose(should_be_identity, identity_batch, atol=1e-5)
    
    def test_rodrigues_determinant(self):
        """Test that rotation matrices have determinant 1"""
        batch_size = 10
        axis_angles = torch.randn(batch_size, 3) * 0.5
        
        rot_mats = batch_rodrigues(axis_angles)
        determinants = torch.det(rot_mats)
        
        assert torch.allclose(determinants, torch.ones(batch_size), atol=1e-5)
    
    def test_rotation_composition(self):
        """Test composition of rotations"""
        # Two small rotations (add batch dimension)
        rot1 = torch.tensor([[0.1, 0.0, 0.0]])  # X-axis rotation
        rot2 = torch.tensor([[0.0, 0.1, 0.0]])  # Y-axis rotation
        
        # Convert to matrices
        mat1 = batch_rodrigues(rot1)[0]  # Remove batch dimension
        mat2 = batch_rodrigues(rot2)[0]  # Remove batch dimension
        
        # Compose rotations
        composed_mat = torch.matmul(mat2, mat1)
        
        # Should still be orthogonal with det=1
        should_be_identity = torch.matmul(composed_mat, composed_mat.T)
        assert torch.allclose(should_be_identity, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(composed_mat), torch.tensor(1.0), atol=1e-5)
    
    def test_large_angle_stability(self):
        """Test stability with large rotation angles"""
        # Large rotation angles
        large_angles = torch.tensor([
            [3.0, 0.0, 0.0],   # ~172 degrees
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.5, 1.5, 1.5]    # Combined rotation
        ])
        
        rot_mats = batch_rodrigues(large_angles)
        
        # Should still produce valid rotation matrices
        batch_size = large_angles.shape[0]
        identity_batch = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        should_be_identity = torch.matmul(rot_mats, rot_mats.transpose(-1, -2))
        
        assert torch.allclose(should_be_identity, identity_batch, atol=1e-4)
        
        determinants = torch.det(rot_mats)
        assert torch.allclose(determinants, torch.ones(batch_size), atol=1e-4)


class TestCoordinateTransforms:
    """Test coordinate system transformations"""
    
    def test_camera_projection_consistency(self):
        """Test 3D to 2D projection consistency"""
        # Mock 3D points
        points_3d = torch.tensor([
            [0.0, 0.0, 1.0],    # On optical axis
            [1.0, 0.0, 2.0],    # Right of center
            [0.0, 1.0, 2.0],    # Above center
        ])
        
        # Camera parameters
        focal = [1000.0, 1000.0]
        princpt = [256.0, 256.0]
        
        # Project to 2D
        x = points_3d[:, 0] / (points_3d[:, 2] + 1e-4) * focal[0] + princpt[0]
        y = points_3d[:, 1] / (points_3d[:, 2] + 1e-4) * focal[1] + princpt[1]
        
        projected_2d = torch.stack([x, y], dim=1)
        
        # Check expected projections (with small epsilon due to floating point precision)
        expected = torch.tensor([
            [256.0, 256.0],     # Center point
            [756.0, 256.0],     # Right shift
            [256.0, 756.0]      # Upward shift
        ])
        
        assert torch.allclose(projected_2d, expected, atol=0.1)  # More tolerant for floating point
    
    def test_depth_scaling(self):
        """Test proper depth scaling in camera projection"""
        # Same 3D point at different depths
        points_3d = torch.tensor([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 1.0, 4.0],
        ])
        
        focal = [1000.0, 1000.0]
        princpt = [256.0, 256.0]
        
        # Project to 2D
        x = points_3d[:, 0] / (points_3d[:, 2] + 1e-4) * focal[0] + princpt[0]
        y = points_3d[:, 1] / (points_3d[:, 2] + 1e-4) * focal[1] + princpt[1]
        
        # Closer points should project further from center
        assert x[0] > x[1] > x[2]  # Decreasing X offset with depth
        assert y[0] > y[1] > y[2]  # Decreasing Y offset with depth
    
    def test_coordinate_range_validation(self):
        """Test coordinate values are in reasonable ranges"""
        # Generate random 3D points in camera space
        batch_size = 100
        points_3d = torch.randn(batch_size, 3)
        points_3d[:, 2] = torch.abs(points_3d[:, 2]) + 0.1  # Ensure positive depth
        
        focal = [1000.0, 1000.0]
        princpt = [256.0, 256.0]
        
        # Project to 2D
        x = points_3d[:, 0] / (points_3d[:, 2] + 1e-4) * focal[0] + princpt[0]
        y = points_3d[:, 1] / (points_3d[:, 2] + 1e-4) * focal[1] + princpt[1]
        
        # Check no NaN or infinite values
        assert torch.all(torch.isfinite(x))
        assert torch.all(torch.isfinite(y))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])