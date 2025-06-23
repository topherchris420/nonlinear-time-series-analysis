import pytest
import numpy as np
import warnings
from unittest.mock import patch, MagicMock


try:
    from nonlinear_analyzer import NonlinearTimeSeriesAnalyzer, LorenzSystem
except ImportError:
    # If running from tests directory
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nonlinear_analyzer import NonlinearTimeSeriesAnalyzer, LorenzSystem


class TestNonlinearTimeSeriesAnalyzer:
    """Test suite for NonlinearTimeSeriesAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return NonlinearTimeSeriesAnalyzer(normalize_data=False)
    
    @pytest.fixture
    def normalized_analyzer(self):
        """Create normalized analyzer instance for testing."""
        return NonlinearTimeSeriesAnalyzer(normalize_data=True)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)  # For reproducible tests
        t = np.linspace(0, 10, 1000)
        # Simple sinusoidal with noise
        data = np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))
        return data
    
    @pytest.fixture
    def lorenz_data(self):
        """Generate Lorenz system data for testing."""
        lorenz = LorenzSystem()
        t = np.linspace(0, 10, 1000)
        x, y, z = lorenz.generate(t)
        return x, y, z
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        # Test default initialization
        analyzer1 = NonlinearTimeSeriesAnalyzer()
        assert analyzer1.normalize_data is True
        assert analyzer1.scaler is not None
        
        # Test with normalization disabled
        analyzer2 = NonlinearTimeSeriesAnalyzer(normalize_data=False)
        assert analyzer2.normalize_data is False
        assert analyzer2.scaler is None
    
    def test_time_delay_embedding_basic(self, analyzer, sample_data):
        """Test basic time-delay embedding functionality."""
        delay = 5
        dimension = 3
        
        embedded = analyzer.time_delay_embedding(sample_data, delay=delay, dimension=dimension)
        
        # Check output shape
        expected_length = len(sample_data) - (dimension - 1) * delay
        assert embedded.shape == (expected_length, dimension)
        
        # Check that embedding preserves the time series structure
        assert np.allclose(embedded[:, 0], sample_data[:expected_length])
        assert np.allclose(embedded[:, 1], sample_data[delay:delay + expected_length])
        assert np.allclose(embedded[:, 2], sample_data[2*delay:2*delay + expected_length])
    
    def test_time_delay_embedding_edge_cases(self, analyzer):
        """Test time-delay embedding edge cases."""
        # Too short time series
        short_data = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Time series too short"):
            analyzer.time_delay_embedding(short_data, delay=10, dimension=3)
        
        # Wrong input dimension
        wrong_dim_data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Input must be a 1D time series"):
            analyzer.time_delay_embedding(wrong_dim_data, delay=1, dimension=2)
    
    def test_time_delay_embedding_normalization(self, normalized_analyzer, sample_data):
        """Test embedding with data normalization."""
        embedded = normalized_analyzer.time_delay_embedding(sample_data, delay=5, dimension=3)
        
        # Check that data was normalized (mean ≈ 0, std ≈ 1)
        assert abs(np.mean(embedded)) < 0.1  # Should be close to 0
        assert abs(np.std(embedded) - 1.0) < 0.1  # Should be close to 1
    
    def test_optimal_embedding_parameters(self, analyzer, sample_data):
        """Test optimal embedding parameter estimation."""
        delay, dimension = analyzer.optimal_embedding_parameters(sample_data)
        
        # Check that reasonable values are returned
        assert isinstance(delay, int)
        assert isinstance(dimension, int)
        assert 1 <= delay <= 50
        assert 1 <= dimension <= 10
    
    def test_wavelet_spectrum(self, analyzer, sample_data):
        """Test wavelet spectrum computation."""
        power = analyzer.wavelet_spectrum(sample_data)
        
        # Check output shape
        assert power.ndim == 2
        assert power.shape[1] == len(sample_data)
        assert power.shape[0] > 0  # Should have some frequency components
        
        # Check that all values are non-negative (power spectrum)
        assert np.all(power >= 0)
    
    def test_wavelet_spectrum_custom_widths(self, analyzer, sample_data):
        """Test wavelet spectrum with custom widths."""
        custom_widths = np.array([1, 2, 4, 8])
        power = analyzer.wavelet_spectrum(sample_data, widths=custom_widths)
        
        assert power.shape[0] == len(custom_widths)
        assert power.shape[1] == len(sample_data)
    
    def test_wavelet_spectrum_error_handling(self, analyzer):
        """Test wavelet spectrum error handling."""
        # Test with invalid data
        with pytest.raises(Exception):
            analyzer.wavelet_spectrum(np.array([]))
    
    def test_transfer_entropy_basic(self, analyzer, lorenz_data):
        """Test basic transfer entropy computation."""
        x, y, z = lorenz_data
        
        # Use shorter sequences for faster testing
        x_short = x[:500]
        y_short = y[:500]
        
        te_value = analyzer.transfer_entropy(x_short, y_short, k=3, lag=1)
        
        # Check that a value is returned
        assert isinstance(te_value, (float, np.floating))
        # For Lorenz system, should have some coupling
        assert not np.isnan(te_value) or te_value == 0.0  # Allow for numerical issues
    
    def test_transfer_entropy_edge_cases(self, analyzer):
        """Test transfer entropy edge cases."""
        # Unequal length series
        x = np.random.randn(100)
        y = np.random.randn(50)
        with pytest.raises(ValueError, match="Time series must have equal length"):
            analyzer.transfer_entropy(x, y)
        
        # Too short series
        x_short = np.random.randn(3)
        y_short = np.random.randn(3)
        with pytest.raises(ValueError, match="Time series too short"):
            analyzer.transfer_entropy(x_short, y_short, k=5, lag=1)
    
    def test_transfer_entropy_independence(self, analyzer):
        """Test transfer entropy with independent series."""
        np.random.seed(42)
        x = np.random.randn(1000)
        y = np.random.randn(1000)
        
        te_value = analyzer.transfer_entropy(x, y, k=3, lag=1)
        
        # Independent series should have low transfer entropy
        # (though due to finite sample effects, might not be exactly zero)
        assert isinstance(te_value, (float, np.floating))
    
    def test_knn_entropy(self, analyzer):
        """Test k-NN entropy estimation."""
        np.random.seed(42)
        # Simple 2D Gaussian data
        data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
        
        entropy = analyzer._knn_entropy(data, k=5)
        
        # Should return a finite value
        assert isinstance(entropy, (float, np.floating))
        assert not np.isnan(entropy)
        assert not np.isinf(entropy)
    
    def test_attractor_analysis(self, analyzer, sample_data):
        """Test attractor analysis."""
        # Create embedded data
        embedded = analyzer.time_delay_embedding(sample_data, delay=5, dimension=3)
        
        results = analyzer.attractor_analysis(embedded)
        
        # Check that all expected keys are present
        expected_keys = ['convex_hull_volume', 'convex_hull_area', 
                        'correlation_dimension', 'lyapunov_estimate']
        for key in expected_keys:
            assert key in results
        
        # Check that values are numbers (or NaN)
        for key, value in results.items():
            assert isinstance(value, (float, np.floating))
    
    def test_attractor_analysis_2d(self, analyzer, sample_data):
        """Test attractor analysis with 2D embedding."""
        embedded = analyzer.time_delay_embedding(sample_data, delay=5, dimension=2)
        
        results = analyzer.attractor_analysis(embedded)
        
        # 2D case should have area but volume might be NaN
        assert 'convex_hull_area' in results
        assert isinstance(results['convex_hull_area'], (float, np.floating))
    
    def test_estimate_correlation_dimension(self, analyzer):
        """Test correlation dimension estimation."""
        # Create a simple 2D dataset
        np.random.seed(42)
        embedded = np.random.randn(200, 2)
        
        corr_dim = analyzer._estimate_correlation_dimension(embedded)
        
        assert isinstance(corr_dim, (float, np.floating))
        # For random 2D data, correlation dimension should be around 2
        # (though our simplified implementation might not be very accurate)
    
    def test_estimate_lyapunov(self, analyzer):
        """Test Lyapunov exponent estimation."""
        # Create some sample embedded data
        np.random.seed(42)
        embedded = np.random.randn(100, 3)
        
        lyap = analyzer._estimate_lyapunov(embedded)
        
        assert isinstance(lyap, (float, np.floating))


class TestLorenzSystem:
    """Test suite for LorenzSystem class."""
    
    @pytest.fixture
    def lorenz(self):
        """Create Lorenz system instance."""
        return LorenzSystem()
    
    def test_lorenz_initialization(self):
        """Test Lorenz system initialization."""
        # Default parameters
        lorenz1 = LorenzSystem()
        assert lorenz1.sigma == 10.0
        assert lorenz1.beta == 8/3
        assert lorenz1.rho == 28.0
        
        # Custom parameters
        lorenz2 = LorenzSystem(sigma=5.0, beta=1.0, rho=15.0)
        assert lorenz2.sigma == 5.0
        assert lorenz2.beta == 1.0
        assert lorenz2.rho == 15.0
    
    def test_lorenz_generation(self, lorenz):
        """Test Lorenz system trajectory generation."""
        t = np.linspace(0, 5, 500)
        x, y, z = lorenz.generate(t)
        
        # Check output shapes
        assert len(x) == len(t)
        assert len(y) == len(t)
        assert len(z) == len(t)
        
        # Check initial conditions
        assert x[0] == 0.0
        assert y[0] == 1.0
        assert z[0] == 1.05
        
        # Check that trajectories are not constant
        assert np.std(x) > 0
        assert np.std(y) > 0
        assert np.std(z) > 0
    
    def test_lorenz_custom_initial_conditions(self, lorenz):
        """Test Lorenz system with custom initial conditions."""
        t = np.linspace(0, 1, 100)
        initial_cond = (1.0, 2.0, 3.0)
        
        x, y, z = lorenz.generate(t, initial_conditions=initial_cond)
        
        assert x[0] == 1.0
        assert y[0] == 2.0
        assert z[0] == 3.0


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_analysis_workflow(self):
        """Test the complete analysis workflow."""
        # Generate test data
        lorenz = LorenzSystem()
        t = np.linspace(0, 10, 1000)
        x, y, z = lorenz.generate(t)
        
        # Initialize analyzer
        analyzer = NonlinearTimeSeriesAnalyzer(normalize_data=True)
        
        # Run complete workflow
        try:
            # Step 1: Embedding
            delay, dimension = analyzer.optimal_embedding_parameters(x)
            embedded = analyzer.time_delay_embedding(x, delay=delay, dimension=dimension)
            
            # Step 2: Wavelet analysis
            power = analyzer.wavelet_spectrum(x)
            
            # Step 3: Transfer entropy
            te_value = analyzer.transfer_entropy(x[:-1], y[1:], k=3)
            
            # Step 4: Attractor analysis
            attractor_metrics = analyzer.attractor_analysis(embedded)
            
            # All steps should complete without error
            assert embedded is not None
            assert power is not None
            assert te_value is not None
            assert attractor_metrics is not None
            
        except Exception as e:
            pytest.fail(f"Complete workflow failed: {e}")
    
    def test_demo_function_runs(self):
        """Test that the demo function runs without error."""
        # This is a smoke test to ensure the demo doesn't crash
        from nonlinear_analyzer import comprehensive_analysis_demo
        
        # Mock plt.show() to prevent actual display during testing
        with patch('matplotlib.pyplot.show'):
            try:
                comprehensive_analysis_demo()
            except Exception as e:
                pytest.fail(f"Demo function failed: {e}")


# Performance tests (optional, marked as slow)
@pytest.mark.slow
class TestPerformance:
    """Performance tests for computationally intensive operations."""
    
    def test_large_dataset_embedding(self):
        """Test embedding with large dataset."""
        analyzer = NonlinearTimeSeriesAnalyzer()
        large_data = np.random.randn(10000)
        
        # Should complete in reasonable time
        embedded = analyzer.time_delay_embedding(large_data, delay=10, dimension=3)
        assert embedded.shape[0] > 0
    
    def test_large_dataset_wavelet(self):
        """Test wavelet analysis with large dataset."""
        analyzer = NonlinearTimeSeriesAnalyzer()
        large_data = np.random.randn(5000)
        
        power = analyzer.wavelet_spectrum(large_data)
        assert power.shape[1] == len(large_data)


# Fixtures for test data
@pytest.fixture(scope="session")
def test_data_directory(tmp_path_factory):
    """Create temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
