"""
Device utility functions for optimal hardware selection
"""

import torch

def get_optimal_device(prefer_gpu=True, verbose=False):
    """
    Get the best available device with intelligent selection
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU over CPU for small workloads
        verbose (bool): Print device selection reasoning
    
    Returns:
        str: Device string ('cuda', 'mps', or 'cpu')
    """
    available_devices = []
    
    # Check CUDA
    if torch.cuda.is_available():
        available_devices.append('cuda')
        if verbose:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    # Check MPS  
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        available_devices.append('mps')
        if verbose:
            print("✅ MPS available: Apple Silicon GPU")
    
    # CPU is always available
    available_devices.append('cpu')
    if verbose:
        print("✅ CPU available")
    
    # Selection logic
    if 'cuda' in available_devices:
        device = 'cuda'
        reason = "CUDA provides best performance for most deep learning workloads"
    elif 'mps' in available_devices and prefer_gpu:
        device = 'mps'
        reason = "MPS recommended for Apple Silicon, good for larger batches"
    else:
        device = 'cpu'
        reason = "CPU selected - may be faster for small batch operations"
    
    if verbose:
        print(f"🎯 Selected device: {device.upper()}")
        print(f"💡 Reason: {reason}")
    
    return device

def get_device_for_workload(batch_size=1, model_size='small'):
    """
    Get optimal device based on workload characteristics
    
    Args:
        batch_size (int): Number of samples to process
        model_size (str): 'small', 'medium', or 'large'
    
    Returns:
        str: Optimal device for the workload
    """
    # For VPoser specifically
    if torch.cuda.is_available():
        return 'cuda'  # CUDA is generally best when available
    
    # For Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS is better for larger batches or models
        if batch_size >= 8 or model_size in ['medium', 'large']:
            return 'mps'
        else:
            # For small batches, CPU might be faster due to GPU overhead
            return 'cpu'
    
    return 'cpu'

def create_device_adaptive_tensor(data, target_device=None):
    """
    Create tensor on optimal device
    
    Args:
        data: Input data (numpy array, list, etc.)
        target_device (str, optional): Specific device to use
    
    Returns:
        torch.Tensor: Tensor on optimal device
    """
    if target_device is None:
        target_device = get_optimal_device()
    
    if isinstance(data, torch.Tensor):
        return data.to(target_device)
    else:
        return torch.tensor(data).to(target_device)

def synchronize_device(device):
    """Synchronize operations on the specified device"""
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        if hasattr(torch, 'mps'):
            torch.mps.synchronize()

def get_device_info():
    """Get information about available devices"""
    info = {
        'cpu': {'available': True, 'name': 'CPU'},
        'cuda': {'available': False, 'name': None},
        'mps': {'available': False, 'name': None}
    }
    
    if torch.cuda.is_available():
        info['cuda']['available'] = True
        info['cuda']['name'] = torch.cuda.get_device_name(0)
        info['cuda']['memory'] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['mps']['available'] = True
        info['mps']['name'] = 'Apple Silicon GPU'
    
    return info