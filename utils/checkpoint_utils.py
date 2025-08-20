"""
Utilities for checkpoint loading, validation, and error handling.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def validate_checkpoint_file(checkpoint_path: Path) -> bool:
    """
    Validate that a checkpoint file exists and is readable.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    if not checkpoint_path.exists():
        logging.error(f"Checkpoint file not found: {checkpoint_path}")
        return False
        
    if not checkpoint_path.is_file():
        logging.error(f"Checkpoint path is not a file: {checkpoint_path}")
        return False
        
    if checkpoint_path.stat().st_size == 0:
        logging.error(f"Checkpoint file is empty: {checkpoint_path}")
        return False
        
    # Try to load checkpoint header to verify it's valid
    try:
        # Just check if it can be loaded without loading full data
        checkpoint_info = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint_info, dict):
            logging.error(f"Checkpoint is not a valid dictionary: {checkpoint_path}")
            return False
    except Exception as e:
        logging.error(f"Cannot load checkpoint file: {checkpoint_path}, error: {e}")
        return False
        
    return True


def load_checkpoint_safely(checkpoint_path: Path, map_location: str = 'cpu') -> Optional[Dict[str, Any]]:
    """
    Safely load checkpoint with comprehensive error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map checkpoint to
        
    Returns:
        Dict containing checkpoint data, or None if loading failed
    """
    if not validate_checkpoint_file(checkpoint_path):
        return None
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Validate checkpoint structure
        if not isinstance(checkpoint, dict):
            logging.error(f"Checkpoint is not a dictionary: {checkpoint_path}")
            return None
            
        if len(checkpoint) == 0:
            logging.error(f"Checkpoint is empty: {checkpoint_path}")
            return None
            
        logging.info(f"Successfully loaded checkpoint: {checkpoint_path}")
        return checkpoint
        
    except Exception as e:
        logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def validate_checkpoint_content(checkpoint: Dict[str, Any], required_keys: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate checkpoint content and structure.
    
    Args:
        checkpoint: Checkpoint dictionary
        required_keys: List of keys that must be present
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    if required_keys is None:
        required_keys = ['epoch', 'network']  # Default required keys
    
    # Check required keys
    for key in required_keys:
        if key not in checkpoint:
            errors.append(f"Missing required key: {key}")
    
    # Validate network state if present
    if 'network' in checkpoint:
        network_state = checkpoint['network']
        
        if not isinstance(network_state, dict):
            errors.append("Network state is not a dictionary")
        elif len(network_state) == 0:
            errors.append("Network state is empty")
        else:
            # Check for invalid parameter values
            invalid_params = []
            for name, param in network_state.items():
                if isinstance(param, torch.Tensor):
                    if torch.isnan(param).any():
                        invalid_params.append(f"{name}: contains NaN")
                    elif torch.isinf(param).any():
                        invalid_params.append(f"{name}: contains Inf")
                    elif param.numel() == 0:
                        invalid_params.append(f"{name}: empty tensor")
            
            if invalid_params:
                errors.extend(invalid_params[:5])  # Limit to first 5 errors
    
    # Validate epoch if present
    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        if not isinstance(epoch, (int, float)):
            errors.append(f"Invalid epoch type: {type(epoch)}")
        elif epoch < 0:
            errors.append(f"Invalid epoch value: {epoch}")
    
    return len(errors) == 0, errors


def load_state_dict_with_error_handling(model: torch.nn.Module, 
                                       state_dict: Dict[str, torch.Tensor],
                                       strict: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load state dict with comprehensive error handling and reporting.
    
    Args:
        model: PyTorch model
        state_dict: State dictionary to load
        strict: Whether to strictly match parameter names
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
        
        # Log results
        if missing_keys:
            logging.warning(f"Missing keys when loading state dict: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                logging.debug(f"Missing keys: {missing_keys}")
            else:
                logging.debug(f"Missing keys (first 10): {missing_keys[:10]}")
                
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                logging.debug(f"Unexpected keys: {unexpected_keys}")
            else:
                logging.debug(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
        
        if not missing_keys and not unexpected_keys:
            logging.info("State dict loaded perfectly - all keys matched")
        
        return missing_keys, unexpected_keys
        
    except Exception as e:
        logging.error(f"Failed to load state dict: {e}")
        raise RuntimeError(f"State dict loading failed: {e}") from e


def get_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """
    Get information about a checkpoint without fully loading it.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    info = {
        'path': str(checkpoint_path),
        'exists': False,
        'size_mb': 0,
        'loadable': False,
        'epoch': None,
        'parameter_count': 0,
        'keys': []
    }
    
    if not checkpoint_path.exists():
        return info
        
    info['exists'] = True
    info['size_mb'] = round(checkpoint_path.stat().st_size / (1024 * 1024), 2)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        info['loadable'] = True
        info['keys'] = list(checkpoint.keys())
        
        if 'epoch' in checkpoint:
            info['epoch'] = checkpoint['epoch']
            
        if 'network' in checkpoint:
            network_state = checkpoint['network']
            if isinstance(network_state, dict):
                info['parameter_count'] = sum(
                    p.numel() for p in network_state.values() 
                    if isinstance(p, torch.Tensor)
                )
                
    except Exception as e:
        logging.warning(f"Could not analyze checkpoint {checkpoint_path}: {e}")
        
    return info


def find_checkpoints(directory: Path, extensions: List[str] = None) -> List[Path]:
    """
    Find all checkpoint files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to look for
        
    Returns:
        List of checkpoint file paths
    """
    if extensions is None:
        extensions = ['.pth', '.pth.tar', '.ckpt', '.pt']
    
    checkpoints = []
    
    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return checkpoints
        
    try:
        for ext in extensions:
            checkpoints.extend(directory.glob(f"**/*{ext}"))
    except Exception as e:
        logging.error(f"Error searching for checkpoints in {directory}: {e}")
        
    return sorted(checkpoints)


def compare_model_checkpoint_compatibility(model: torch.nn.Module, 
                                         checkpoint_path: Path) -> Dict[str, Any]:
    """
    Compare model architecture with checkpoint to check compatibility.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint
        
    Returns:
        Dictionary with compatibility information
    """
    compatibility = {
        'compatible': False,
        'matching_keys': 0,
        'missing_keys': 0,
        'unexpected_keys': 0,
        'total_model_params': 0,
        'total_checkpoint_params': 0,
        'parameter_size_matches': 0
    }
    
    checkpoint = load_checkpoint_safely(checkpoint_path)
    if checkpoint is None:
        return compatibility
        
    if 'network' not in checkpoint:
        return compatibility
        
    model_state = model.state_dict()
    checkpoint_state = checkpoint['network']
    
    model_keys = set(model_state.keys())
    checkpoint_keys = set(checkpoint_state.keys())
    
    matching_keys = model_keys.intersection(checkpoint_keys)
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    compatibility.update({
        'matching_keys': len(matching_keys),
        'missing_keys': len(missing_keys),
        'unexpected_keys': len(unexpected_keys),
        'total_model_params': sum(p.numel() for p in model_state.values()),
        'total_checkpoint_params': sum(
            p.numel() for p in checkpoint_state.values() 
            if isinstance(p, torch.Tensor)
        )
    })
    
    # Check parameter size matches for common keys
    size_matches = 0
    for key in matching_keys:
        if isinstance(checkpoint_state[key], torch.Tensor) and isinstance(model_state[key], torch.Tensor):
            if checkpoint_state[key].shape == model_state[key].shape:
                size_matches += 1
                
    compatibility['parameter_size_matches'] = size_matches
    
    # Consider compatible if we have substantial overlap
    compatibility['compatible'] = (
        len(matching_keys) > 0 and 
        len(matching_keys) >= len(missing_keys) * 0.5 and  # At least 50% coverage
        size_matches == len(matching_keys)  # All matching keys have correct sizes
    )
    
    return compatibility