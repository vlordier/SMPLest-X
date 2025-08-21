"""
Comprehensive pytest suite for SMPLest-X model validation and testing.
Tests model architecture, forward pass, loss computation, and parameter handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Force CPU for testing to avoid MPS-specific issues
os.environ['DEVICE'] = 'cpu'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.SMPLest_X import Model
from models.module import TransformerDecoderHead, ViT
from models.loss import CoordLoss, ParamLoss
from human_models.pytorch3d_smplx import Direct_SMPLX
from human_models.human_models import SMPLX
from utils.transforms import rot6d_to_axis_angle, batch_rodrigues

# Reset singletons to ensure they use CPU device setting
Direct_SMPLX.reset_instance()
SMPLX.reset_instance()


class MockConfig:
    """Mock configuration for testing"""
    def __init__(self):
        self.model = MockModelConfig()
        self.train = MockTrainConfig()
        self.data = MockDataConfig()

class MockModelConfig:
    def __init__(self):
        self.human_model_path = "./human_models/human_model_files"
        self.focal = [5000.0, 5000.0]
        self.princpt = [96.0, 128.0]  
        self.camera_3d_size = 2.5
        self.input_body_shape = [256, 192]
        self.output_hm_shape = [16, 16, 12]
        self.encoder_config = {
            'img_size': (256, 192),
            'patch_size': 16,
            'embed_dim': 1280,
            'depth': 32,
            'num_heads': 16,
            'task_tokens_num': 80,
            'ratio': 1,
            'use_checkpoint': False,
            'mlp_ratio': 4,
            'qkv_bias': True,
            'drop_path_rate': 0.55
        }
        self.decoder_config = {
            'feat_dim': 1280,
            'dim_out': 512, 
            'task_tokens_num': 80
        }

class MockTrainConfig:
    def __init__(self):
        self.smplx_kps_3d_weight = 1.0
        self.smplx_kps_2d_weight = 1.0
        self.smplx_pose_weight = 1.0
        self.smplx_shape_weight = 1.0
        self.smplx_orient_weight = 1.0
        self.smplx_hand_kps_3d_weight = 1.0
        self.hand_root_weight = 1.0
        self.hand_loss = True
        self.no_chain_hand_loss = False

class MockDataConfig:
    def __init__(self):
        self.testset = 'test'
        self.bbox_ratio = 1.25


class TestModelArchitecture:
    """Test model architecture and initialization"""
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        config = MockConfig()
        
        # Mock encoder/decoder for testing
        encoder = ViT(**config.model.encoder_config)
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        
        # Test model creation
        model = Model(config, encoder, decoder)
        
        assert model is not None
        assert hasattr(model, 'smpl_x')
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'decoder')
        assert hasattr(model, 'coord_loss')
        assert hasattr(model, 'param_loss')

    def test_encoder_architecture(self):
        """Test Vision Transformer encoder"""
        config = MockConfig()
        encoder = ViT(**config.model.encoder_config)
        
        # Test input/output shapes
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 256, 192)
        
        with torch.no_grad():
            img_feat, task_tokens = encoder(input_tensor)
            
        assert img_feat.shape[0] == batch_size
        assert task_tokens.shape == (batch_size, 80, 1280)  # [B, task_tokens, embed_dim]
        
    def test_decoder_architecture(self):
        """Test Transformer decoder head"""
        config = MockConfig()
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        
        batch_size = 2
        task_tokens = torch.randn(batch_size, 80, 1280)
        img_feat = torch.randn(batch_size, 1280, 16, 12)  # Feature map from encoder
        
        with torch.no_grad():
            pred_params = decoder(task_tokens, img_feat)
            
        # Check all expected outputs
        expected_keys = [
            'body_root_pose', 'body_pose', 'body_betas', 'body_cam',
            'lhand_root_pose', 'rhand_root_pose', 'lhand_pose', 'rhand_pose',
            'lhand_cam', 'rhand_cam', 'face_root_pose', 'face_expression', 
            'face_jaw_pose', 'face_cam'
        ]
        
        for key in expected_keys:
            assert key in pred_params, f"Missing key: {key}"
            assert pred_params[key].shape[0] == batch_size


class TestModelForwardPass:
    """Test model forward pass and output validation"""
    
    @pytest.fixture
    def model_setup(self):
        """Setup model for testing"""
        config = MockConfig()
        encoder = ViT(**config.model.encoder_config)
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        model = Model(config, encoder, decoder)
        # Force model to CPU to avoid MPS-specific issues in testing
        model = model.cpu()
        return model, config
    
    def test_forward_pass_test_mode(self, model_setup):
        """Test forward pass in test mode"""
        model, config = model_setup
        batch_size = 2
        
        # Mock inputs (force to CPU for testing)
        inputs = {
            'img': torch.randn(batch_size, 3, 256, 192).cpu()
        }
        
        targets = {}
        meta_info = {}
        
        with torch.no_grad():
            output = model(inputs, targets, meta_info, mode='test')
            
        # Check essential outputs
        expected_keys = [
            'smplx_joint_proj', 'smplx_mesh_cam', 'smplx_root_pose',
            'smplx_body_pose', 'smplx_lhand_pose', 'smplx_rhand_pose',
            'smplx_jaw_pose', 'smplx_shape', 'smplx_expr', 'cam_trans'
        ]
        
        for key in expected_keys:
            assert key in output, f"Missing output key: {key}"
            assert output[key].shape[0] == batch_size
            
        # Check specific shapes
        assert output['smplx_mesh_cam'].shape == (batch_size, 10475, 3)  # SMPL-X vertices
        assert output['smplx_joint_proj'].shape[1] == 137  # SMPL-X joints
        assert output['smplx_shape'].shape == (batch_size, 10)  # Beta parameters
        assert output['smplx_expr'].shape == (batch_size, 10)  # Expression parameters

    def test_camera_transformation(self, model_setup):
        """Test camera parameter transformation"""
        model, config = model_setup
        batch_size = 3
        
        # Test camera parameter conversion
        cam_param = torch.randn(batch_size, 3)
        
        with torch.no_grad():
            cam_trans = model.get_camera_trans(cam_param)
            
        assert cam_trans.shape == (batch_size, 3)
        assert torch.all(cam_trans[:, 2] > 0), "Camera Z translation should be positive"

    def test_coordinate_generation(self, model_setup):
        """Test 3D coordinate generation from SMPL-X parameters"""
        model, config = model_setup
        batch_size = 2
        
        # Mock SMPL-X parameters
        root_pose = torch.randn(batch_size, 3)
        body_pose = torch.randn(batch_size, 63)  # 21 joints * 3
        lhand_pose = torch.randn(batch_size, 45)  # 15 joints * 3
        rhand_pose = torch.randn(batch_size, 45)
        jaw_pose = torch.randn(batch_size, 3)
        shape = torch.randn(batch_size, 10)
        expr = torch.randn(batch_size, 10)
        cam_trans = torch.randn(batch_size, 3)
        
        with torch.no_grad():
            joint_proj, joint_cam, joint_cam_wo_ra, mesh_cam, root_cam = model.get_coord(
                root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose, 
                shape, expr, cam_trans, mode='test'
            )
            
        # Check output shapes
        assert joint_proj.shape == (batch_size, 137, 2)  # 2D projections
        assert joint_cam.shape == (batch_size, 137, 3)   # 3D coordinates with root alignment
        assert joint_cam_wo_ra.shape == (batch_size, 137, 3)  # 3D without root alignment
        assert mesh_cam.shape == (batch_size, 10475, 3)  # SMPL-X mesh vertices
        assert root_cam.shape == (batch_size, 1, 3)      # Root joint position


class TestLossFunctions:
    """Test loss function implementations"""
    
    def test_coordinate_loss(self):
        """Test coordinate loss function"""
        coord_loss = CoordLoss()
        batch_size = 4
        num_joints = 137
        
        # Mock data
        pred_coords = torch.randn(batch_size, num_joints, 3)
        gt_coords = torch.randn(batch_size, num_joints, 3)
        valid_mask = torch.ones(batch_size, num_joints, 3)
        
        loss = coord_loss(pred_coords, gt_coords, valid_mask)
        
        assert loss.shape == (batch_size, num_joints, 3)
        assert torch.all(loss >= 0), "Loss should be non-negative"
        
        # Test with 3D mask
        is_3D = torch.ones(batch_size)
        loss_3d = coord_loss(pred_coords, gt_coords, valid_mask, is_3D)
        assert loss_3d.shape == (batch_size, num_joints, 3)

    def test_parameter_loss(self):
        """Test parameter loss function"""
        param_loss = ParamLoss()
        batch_size = 3
        param_dim = 10
        
        # Mock data
        pred_params = torch.randn(batch_size, param_dim)
        gt_params = torch.randn(batch_size, param_dim)
        valid_mask = torch.ones(batch_size, param_dim)
        
        loss = param_loss(pred_params, gt_params, valid_mask)
        
        assert loss.shape == (batch_size, param_dim)
        assert torch.all(loss >= 0), "Loss should be non-negative"


class TestRotationTransforms:
    """Test rotation representation transformations"""
    
    def test_rot6d_to_axis_angle(self):
        """Test 6D rotation to axis-angle conversion"""
        batch_size = 5
        num_joints = 10
        
        # Mock 6D rotation representations
        rot6d = torch.randn(batch_size * num_joints, 6)
        
        axis_angle = rot6d_to_axis_angle(rot6d)
        axis_angle = axis_angle.view(batch_size, num_joints, 3)
        
        assert axis_angle.shape == (batch_size, num_joints, 3)
        
        # Check angle magnitude is reasonable (should be < 2π)
        angles = torch.norm(axis_angle, dim=-1)
        assert torch.all(angles < 2 * np.pi), "Rotation angles should be < 2π"

    def test_batch_rodrigues(self):
        """Test batch Rodrigues formula (axis-angle to rotation matrix)"""
        batch_size = 3
        
        # Mock axis-angle rotations
        axis_angle = torch.randn(batch_size, 3) * 0.5  # Small rotations
        
        rot_mat = batch_rodrigues(axis_angle)
        
        assert rot_mat.shape == (batch_size, 3, 3)
        
        # Check if rotation matrices are orthogonal (R @ R.T = I)
        identity = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        should_be_identity = torch.matmul(rot_mat, rot_mat.transpose(-1, -2))
        
        assert torch.allclose(should_be_identity, identity, atol=1e-5), \
            "Rotation matrices should be orthogonal"
        
        # Check determinant is 1 (proper rotation)
        det = torch.det(rot_mat)
        assert torch.allclose(det, torch.ones(batch_size), atol=1e-5), \
            "Rotation matrix determinant should be 1"


class TestModelValidation:
    """Test model validation and error handling"""
    
    @pytest.fixture
    def model_setup(self):
        """Setup model for testing"""
        config = MockConfig()
        encoder = ViT(**config.model.encoder_config)
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        model = Model(config, encoder, decoder)
        # Force model to CPU to avoid MPS-specific issues in testing
        model = model.cpu()
        return model, config
    
    def test_input_shape_validation(self, model_setup):
        """Test model handles different input shapes correctly"""
        model, config = model_setup
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            inputs = {'img': torch.randn(batch_size, 3, 256, 192)}
            
            with torch.no_grad():
                output = model(inputs, {}, {}, mode='test')
                
            assert output['smplx_mesh_cam'].shape[0] == batch_size

    def test_gradient_flow(self, model_setup):
        """Test gradient flow through the model"""
        model, config = model_setup
        model.train()
        
        batch_size = 2
        inputs = {'img': torch.randn(batch_size, 3, 256, 192)}
        
        # Create mock targets for training
        targets = {
            'smplx_pose': torch.randn(batch_size, 159),  # 3+63+45+45+3
            'smplx_shape': torch.randn(batch_size, 10),
            'smplx_expr': torch.randn(batch_size, 10),
            'smplx_cam_trans': torch.randn(batch_size, 3),
            'joint_cam': torch.randn(batch_size, 137, 3),
            'smplx_joint_cam': torch.randn(batch_size, 137, 3),
            'joint_img': torch.randn(batch_size, 137, 3),
            'lhand_root': torch.randn(batch_size, 3),
            'rhand_root': torch.randn(batch_size, 3)
        }
        
        # Create mock meta_info
        meta_info = {
            'smplx_pose_valid': torch.ones(batch_size, 159),
            'smplx_shape_valid': torch.ones(batch_size),
            'smplx_expr_valid': torch.ones(batch_size),
            'joint_trunc': torch.ones(batch_size, 137, 3),
            'is_3D': torch.ones(batch_size)
        }
        
        # Forward pass
        loss_dict = model(inputs, targets, meta_info, mode='train')
        
        # Check loss components exist
        expected_losses = [
            'smplx_orient', 'smplx_pose', 'smplx_shape', 'smplx_expr',
            'joint_cam', 'smplx_joint_cam', 'joint_proj', 'hand_root'
        ]
        
        for loss_name in expected_losses:
            assert loss_name in loss_dict, f"Missing loss: {loss_name}"
            assert isinstance(loss_dict[loss_name], torch.Tensor)
            
        # Test gradient computation
        total_loss = sum(loss_dict.values())
        total_loss.backward()
        
        # Check some gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break
                
        assert has_gradients, "Model should have gradients after backward pass"

    def test_model_device_consistency(self, model_setup):
        """Test model works consistently across different devices"""
        model, config = model_setup
        
        # Test on CPU
        inputs = {'img': torch.randn(1, 3, 256, 192)}
        
        with torch.no_grad():
            output_cpu = model(inputs, {}, {}, mode='test')
            
        assert output_cpu['smplx_mesh_cam'].device.type == 'cpu'
        
        # Test GPU if available
        from utils.device_utils import get_device, to_device
        device = get_device()
        if device.type != 'cpu':
            model_gpu = to_device(model)
            inputs_gpu = {'img': to_device(torch.randn(1, 3, 512, 384))}
            
            with torch.no_grad():
                output_gpu = model_gpu(inputs_gpu, {}, {}, mode='test')
                
            assert output_gpu['smplx_mesh_cam'].device.type == device.type


class TestModelIntegration:
    """Integration tests for the complete model pipeline"""
    
    def test_inference_pipeline_consistency(self):
        """Test that inference pipeline produces consistent outputs"""
        config = MockConfig()
        encoder = ViT(**config.model.encoder_config)
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        model = Model(config, encoder, decoder)
        model.eval()
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        inputs = {'img': torch.randn(1, 3, 256, 192)}
        
        with torch.no_grad():
            output1 = model(inputs, {}, {}, mode='test')
            
        # Same input should give same output
        torch.manual_seed(42)
        inputs2 = {'img': torch.randn(1, 3, 512, 384)}
        
        with torch.no_grad():
            output2 = model(inputs2, {}, {}, mode='test')
            
        # Check outputs are identical
        for key in output1.keys():
            if isinstance(output1[key], torch.Tensor):
                assert torch.allclose(output1[key], output2[key], atol=1e-6), \
                    f"Outputs should be identical for key: {key}"

    def test_model_memory_efficiency(self):
        """Test model memory usage is reasonable"""
        config = MockConfig()
        encoder = ViT(**config.model.encoder_config)
        decoder = TransformerDecoderHead(**config.model.decoder_config, human_model_path=config.model.human_model_path)
        model = Model(config, encoder, decoder)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Model should have reasonable number of parameters (not too large)
        assert total_params < 1e9, f"Model has too many parameters: {total_params}"
        assert trainable_params > 1e6, f"Model should have sufficient trainable parameters: {trainable_params}"
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    # Run specific tests for development
    pytest.main([__file__, "-v", "--tb=short"])