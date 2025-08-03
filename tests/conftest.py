"""
Pytest configuration and shared fixtures for all tests.
"""
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import torch


# Configure asyncio for tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for tests."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def mock_models_dir(temp_dir):
    """Create a mock models directory."""
    models_path = temp_dir / "models"
    models_path.mkdir(exist_ok=True)
    return str(models_path)


@pytest.fixture
def mock_audio_tensor():
    """Create a sample audio tensor for testing."""
    sample_rate = 16000
    duration = 1.0  # 1 second
    samples = int(sample_rate * duration)
    audio = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, samples))
    return audio, sample_rate


@pytest.fixture
def mock_app_state():
    """Create a mock FastAPI app state."""
    state = Mock()
    state.generator = Mock()
    state.voice_cloning_system = Mock()
    state.voice_enhancement_system = Mock()
    state.voice_memory = Mock()
    state.model_name = "csm-1b"
    state.voices = {
        "alloy": 42,
        "echo": 43,
        "fable": 44,
        "onyx": 45,
        "nova": 46,
        "shimmer": 47
    }
    return state


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton resets here if needed
    yield


@pytest.fixture
def disable_logging():
    """Disable logging during tests to reduce noise."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Markers for different test types
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "asyncio: mark test as async")


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip GPU tests when CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)