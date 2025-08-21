"""
VPoser integration utilities for SMPLest-X
Provides pose prior regularization using VPoser latent codes
"""

import torch
import torch.nn as nn
import numpy as np
import os
import logging
from typing import Dict, Optional, Tuple
from .device_utils import to_device

try:
    from human_body_prior.tools.model_loader import load_model
    from human_body_prior.models.vposer_model import VPoser
    VPOSER_AVAILABLE = True
except ImportError:
    VPOSER_AVAILABLE = False
    logging.warning("VPoser not available. Install human-body-prior package.")


class VPoserWrapper(nn.Module):
    """
    Wrapper for VPoser integration with SMPLest-X
    Provides pose encoding/decoding and pose prior regularization
    """
    
    def __init__(self, vposer_ckpt_dir: str, device: str = 'cuda'):
        super(VPoserWrapper, self).__init__()
        
        if not VPOSER_AVAILABLE:
            raise ImportError("VPoser not available. Install human-body-prior package.")
            
        self.device = device
        self.vposer_ckpt_dir = vposer_ckpt_dir
        self.latent_dim = 32  # VPoser latent dimension
        
        # Load VPoser model
        self._load_vposer_model()
        
    def _load_vposer_model(self):
        """Load pre-trained VPoser model"""
        if not os.path.exists(self.vposer_ckpt_dir):
            raise FileNotFoundError(f"VPoser checkpoint directory not found: {self.vposer_ckpt_dir}")
            
        try:
            self.vposer, _ = load_model(
                self.vposer_ckpt_dir,
                model_code=VPoser,
                remove_words_in_model_weights='vp_model.',
                disable_grad=True
            )
            self.vposer = to_device(self.vposer, self.device)
            self.vposer.eval()
            logging.info(f"VPoser model loaded from {self.vposer_ckpt_dir}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load VPoser model: {e}")
    
    def encode_pose(self, body_pose: torch.Tensor) -> torch.Tensor:
        """
        Encode SMPL body pose to VPoser latent codes (poZ)
        
        Args:
            body_pose: SMPL body pose parameters [batch_size, 63]
            
        Returns:
            pose_latent: VPoser latent codes [batch_size, 32]
        """
        with torch.no_grad():
            pose_latent = self.vposer.encode(body_pose).mean
        return pose_latent
    
    def decode_pose(self, pose_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode VPoser latent codes to SMPL body pose
        
        Args:
            pose_latent: VPoser latent codes [batch_size, 32]
            
        Returns:
            body_pose: SMPL body pose parameters [batch_size, 63]
        """
        decoded = self.vposer.decode(pose_latent)
        body_pose = decoded['pose_body'].contiguous().view(-1, 63)
        return body_pose
    
    def sample_poses(self, batch_size: int) -> torch.Tensor:
        """
        Sample random poses from VPoser prior
        
        Args:
            batch_size: Number of poses to sample
            
        Returns:
            body_pose: Sampled SMPL body pose parameters [batch_size, 63]
        """
        # Sample from standard normal distribution
        pose_latent = to_device(torch.randn(batch_size, self.latent_dim), self.device)
        return self.decode_pose(pose_latent)
    
    def compute_pose_prior_loss(self, pose_latent: torch.Tensor, 
                               weight: float = 1.0) -> torch.Tensor:
        """
        Compute pose prior regularization loss
        Encourages pose latent codes to follow standard normal distribution
        
        Args:
            pose_latent: VPoser latent codes [batch_size, 32]
            weight: Loss weight
            
        Returns:
            loss: Pose prior loss scalar
        """
        # L2 loss to encourage latent codes to be close to zero (standard normal)
        pose_prior_loss = torch.mean(pose_latent.pow(2))
        return weight * pose_prior_loss
    
    def regularize_pose(self, body_pose: torch.Tensor, 
                       alpha: float = 0.1) -> torch.Tensor:
        """
        Regularize pose using VPoser encode-decode cycle
        
        Args:
            body_pose: Input SMPL body pose [batch_size, 63]
            alpha: Regularization strength (0.0 = no regularization, 1.0 = full)
            
        Returns:
            regularized_pose: Regularized SMPL body pose [batch_size, 63]
        """
        if alpha == 0.0:
            return body_pose
            
        # Encode to latent space and decode back
        pose_latent = self.encode_pose(body_pose)
        regularized_pose = self.decode_pose(pose_latent)
        
        # Blend original and regularized pose
        return (1 - alpha) * body_pose + alpha * regularized_pose


class VPoserLoss(nn.Module):
    """
    VPoser-based loss functions for pose regularization
    """
    
    def __init__(self, vposer_wrapper: VPoserWrapper):
        super(VPoserLoss, self).__init__()
        self.vposer = vposer_wrapper
        
    def forward(self, body_pose: torch.Tensor, 
                pose_prior_weight: float = 1.0,
                reconstruction_weight: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        Compute VPoser-based losses
        
        Args:
            body_pose: Predicted SMPL body pose [batch_size, 63]
            pose_prior_weight: Weight for pose prior loss
            reconstruction_weight: Weight for pose reconstruction loss
            
        Returns:
            losses: Dictionary of loss components
        """
        # Encode pose to latent space
        pose_latent = self.vposer.encode_pose(body_pose)
        
        # Decode back to pose space
        reconstructed_pose = self.vposer.decode_pose(pose_latent)
        
        # Pose prior loss (encourage normal distribution)
        pose_prior_loss = self.vposer.compute_pose_prior_loss(pose_latent, pose_prior_weight)
        
        # Pose reconstruction loss (cycle consistency)
        pose_recon_loss = reconstruction_weight * torch.mean((body_pose - reconstructed_pose).pow(2))
        
        return {
            'pose_prior_loss': pose_prior_loss,
            'pose_reconstruction_loss': pose_recon_loss,
            'total_vposer_loss': pose_prior_loss + pose_recon_loss
        }


def download_vposer_model(download_dir: str = './data/vposer_v1_0') -> str:
    """
    Download VPoser model if not available locally
    
    Args:
        download_dir: Directory to download VPoser model
        
    Returns:
        vposer_ckpt_dir: Path to VPoser checkpoint directory
    """
    # Note: This would require implementation of download logic
    # For now, user needs to manually download from SMPL-X website
    vposer_ckpt_dir = download_dir  # VPoser expects the main directory, not snapshots subdir
    
    if not os.path.exists(os.path.join(vposer_ckpt_dir, 'snapshots')):
        raise FileNotFoundError(
            f"VPoser model not found at {vposer_ckpt_dir}/snapshots. "
            "Please download VPoser model from https://smpl-x.is.tue.mpg.de/ "
            "and extract to the specified directory."
        )
    
    return vposer_ckpt_dir


def create_vposer_wrapper(vposer_ckpt_dir: Optional[str] = None, 
                         device: str = 'cuda') -> Optional[VPoserWrapper]:
    """
    Factory function to create VPoser wrapper with error handling
    
    Args:
        vposer_ckpt_dir: Path to VPoser checkpoint directory
        device: Device to load VPoser on
        
    Returns:
        vposer_wrapper: VPoser wrapper instance or None if failed
    """
    if not VPOSER_AVAILABLE:
        logging.warning("VPoser not available - proceeding without pose prior")
        return None
        
    if vposer_ckpt_dir is None:
        vposer_ckpt_dir = './data/vposer_v1_0'
        
    try:
        return VPoserWrapper(vposer_ckpt_dir, device)
    except Exception as e:
        logging.warning(f"Failed to initialize VPoser: {e}. Proceeding without pose prior.")
        return None