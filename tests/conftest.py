"""
pytest configuration and fixtures for SMPLest-X tests
"""

import pytest
import torch
import sys
from pathlib import Path
from utils.device_utils import get_device

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
    return get_device()

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