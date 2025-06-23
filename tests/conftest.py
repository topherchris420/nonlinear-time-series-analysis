import pytest
import numpy as np

@pytest.fixture(scope="session")
def lorenz_data_large():
    """Large Lorenz dataset for performance tests."""
    # Generate once per test session
    pass
