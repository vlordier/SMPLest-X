"""Device utility functions for cross-platform compatibility."""

import torch


def get_device():
    """
    Get the best available device for PyTorch operations.
    
    Returns:
        torch.device: The best available device (MPS on Mac, CUDA on GPU machines, CPU otherwise)
    """
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(tensor_or_model, device=None):
    """
    Move tensor or model to the specified device.
    
    Args:
        tensor_or_model: Tensor or model to move
        device: Target device (if None, uses get_device())
        
    Returns:
        Tensor or model moved to the target device
    """
    if device is None:
        device = get_device()
    return tensor_or_model.to(device)


def is_cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def is_mps_available():
    """Check if MPS is available."""
    return torch.backends.mps.is_available()


def synchronize():
    """Synchronize the current device."""
    if torch.cuda.is_available() and torch.cuda.current_device() >= 0:
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_cache():
    """Empty the cache for the current device."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_device_count():
    """Get the number of available devices."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1  # MPS typically has one device
    else:
        return 1  # CPU


def set_device(device_id):
    """Set the current device."""
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    # MPS doesn't need explicit device setting


def get_device_name():
    """Get a string representation of the current device."""
    device = get_device()
    return device.type


def setup_mac_environment():
    """Setup Mac-specific environment variables for optimal performance."""
    import os
    
    # Enable MPS fallback for unsupported operations
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    
    # Prevent OpenMP threading issues on Mac
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    
    # MPS memory management
    os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
    
    # Disable problematic optimizations
    os.environ.setdefault('PYTORCH_MPS_ENABLE_GRAPH_OPTIMIZATION', '0')
    
    return True


def safe_to_device(tensor_or_model, device=None, fallback_cpu=True):
    """Safely move tensor/model to device with fallback on Mac."""
    if device is None:
        device = get_device()
    
    try:
        return tensor_or_model.to(device)
    except RuntimeError as e:
        if fallback_cpu and 'mps' in device.type.lower():
            print(f"⚠️  MPS error, falling back to CPU: {e}")
            return tensor_or_model.to('cpu')
        else:
            raise


def handle_mps_fallback(func, *args, **kwargs):
    """Execute function with MPS fallback to CPU if needed."""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if 'mps' in str(e).lower() or 'metal' in str(e).lower():
            print(f"⚠️  MPS operation failed, using CPU fallback")
            # Move tensors to CPU
            cpu_args = []
            for arg in args:
                if hasattr(arg, 'cpu'):
                    cpu_args.append(arg.cpu())
                else:
                    cpu_args.append(arg)
            
            result = func(*cpu_args, **kwargs)
            
            # Try to move result back to MPS
            if hasattr(result, 'to'):
                try:
                    return result.to(get_device())
                except:
                    return result
            return result
        else:
            raise
