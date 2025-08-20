"""
pytest configuration and fixtures for SMPLest-X tests
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set random seeds for reproducible tests
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Set torch settings for testing
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    yield
    
    # Cleanup after tests
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def device():
    """Return the best available device"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

@pytest.fixture
def batch_sizes():
    """Common batch sizes for testing"""
    return [1, 2, 4]

@pytest.fixture
def image_shapes():
    """Common image shapes for testing"""
    return [
        (3, 224, 224),
        (3, 384, 384),
        (3, 512, 384)
    ]