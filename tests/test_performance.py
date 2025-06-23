import time
import pytest

def test_embedding_performance():
    """Test that embedding completes in reasonable time."""
    start = time.time()
    # ... run embedding
    duration = time.time() - start
    assert duration < 5.0  # Should complete in under 5 seconds
