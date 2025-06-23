import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, ricker
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
from typing import Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NonlinearTimeSeriesAnalyzer:
    """
    A comprehensive toolkit for nonlinear time series analysis including:
    - Time-delay embedding
    - Wavelet analysis
    - Transfer entropy estimation
    - Attractor reconstruction and analysis
    """
    
    def __init__(self, normalize_data: bool = True):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        normalize_data : bool
            Whether to normalize input data for better numerical stability
        """
        self.normalize_data = normalize_data
        self.scaler = StandardScaler() if normalize_data else None
        
    def time_delay_embedding(self, x: np.ndarray, delay: int = 10, 
                           dimension: int = 3) -> np.ndarray:
        """
        Perform time-delay embedding using Takens' theorem.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        delay : int
            Time delay (tau) for embedding
        dimension : int
            Embedding dimension
            
        Returns:
        --------
        np.ndarray
            Embedded time series of shape (n_points, dimension)
        """
        if len(x.shape) != 1:
            raise ValueError("Input must be a 1D time series")
        
        n_points = len(x) - (dimension - 1) * delay
        if n_points <= 0:
            raise ValueError("Time series too short for given delay and dimension")
        
        # Normalize if requested
        if self.normalize_data:
            x = self.scaler.fit_transform(x.reshape(-1, 1)).flatten()
        
        embedded = np.empty((n_points, dimension))
        for i in range(dimension):
            embedded[:, i] = x[i * delay : i * delay + n_points]
        
        logger.info(f"Time-delay embedding completed: {embedded.shape}")
        return embedded
    
    def optimal_embedding_parameters(self, x: np.ndarray, 
                                   max_delay: int = 50,
                                   max_dimension: int = 10) -> Tuple[int, int]:
        """
        Estimate optimal embedding parameters using mutual information and FNN.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        max_delay : int
            Maximum delay to consider
        max_dimension : int
            Maximum dimension to consider
            
        Returns:
        --------
        Tuple[int, int]
            Optimal (delay, dimension)
        """
        # Simplified implementation - in practice, would use mutual information
        # and false nearest neighbors algorithms
        
        # For demonstration, use autocorrelation for delay estimation
        delays = np.arange(1, max_delay + 1)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # Find first minimum or where autocorr drops below 1/e
        optimal_delay = np.argmax(autocorr < 1/np.e) + 1
        optimal_delay = max(1, min(optimal_delay, max_delay))
        
        # Use a heuristic for dimension (would use FNN in practice)
        optimal_dimension = min(3, max_dimension)
        
        logger.info(f"Estimated optimal parameters: delay={optimal_delay}, dimension={optimal_dimension}")
        return optimal_delay, optimal_dimension
    
    def wavelet_spectrum(self, x: np.ndarray, 
                        widths: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute continuous wavelet transform using Ricker wavelet.
        
        Parameters:
        -----------
        x : np.ndarray
            Input time series
        widths : np.ndarray, optional
            Wavelet scales to use
            
        Returns:
        --------
        np.ndarray
            Wavelet power spectrum
        """
        if widths is None:
            widths = np.logspace(0, 2, 50)  # Logarithmic spacing for better resolution
        
        try:
            coef = cwt(x, ricker, widths)
            power = np.abs(coef) ** 2
            logger.info(f"Wavelet spectrum computed: {power.shape}")
            return power
        except Exception as e:
            logger.error(f"Wavelet computation failed: {e}")
            raise
    
    def transfer_entropy(self, x: np.ndarray, y: np.ndarray, 
                        k: int = 5, lag: int = 1) -> float:
        """
        Estimate transfer entropy using k-nearest neighbors.
        
        Parameters:
        -----------
        x : np.ndarray
            Source time series
        y : np.ndarray
            Target time series
        k : int
            Number of nearest neighbors
        lag : int
            Time lag for causality
            
        Returns:
        --------
        float
            Transfer entropy from x to y
        """
        if len(x) != len(y):
            raise ValueError("Time series must have equal length")
        
        if len(x) < k + lag + 1:
            raise ValueError("Time series too short for given parameters")
        
        # Prepare lagged versions
        n = len(x) - lag
        x_past = x[:-lag].reshape(-1, 1)
        y_past = y[:-lag].reshape(-1, 1)
        y_future = y[lag:].reshape(-1, 1)
        
        # Construct joint spaces
        y_past_future = np.hstack([y_past, y_future])
        x_y_past_future = np.hstack([x_past, y_past, y_future])
        
        try:
            # Estimate conditional mutual information using k-NN
            te = self._knn_entropy(x_y_past_future, k) + self._knn_entropy(y_past, k) - \
                 self._knn_entropy(y_past_future, k) - self._knn_entropy(np.hstack([x_past, y_past]), k)
            
            logger.info(f"Transfer entropy computed: {te:.6f}")
            return te
        except Exception as e:
            logger.error(f"Transfer entropy computation failed: {e}")
            return np.nan
    
    def _knn_entropy(self, data: np.ndarray, k: int) -> float:
        """
        Estimate entropy using k-nearest neighbors.
        
        Parameters:
        -----------
        data : np.ndarray
            Data points
        k : int
            Number of nearest neighbors
            
        Returns:
        --------
        float
            Estimated entropy
        """
        n, d = data.shape
        if n <= k:
            return 0.0
        
        # Fit k-NN model
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nbrs.fit(data)
        
        # Get distances to k-th nearest neighbor
        distances, _ = nbrs.kneighbors(data)
        rho = distances[:, -1]  # Distance to k-th neighbor
        
        # Avoid log(0)
        rho = np.maximum(rho, 1e-15)
        
        # Entropy estimation
        entropy = (d * np.mean(np.log(rho)) + 
                  np.log(n - 1) - 
                  np.log(k) + 
                  d * np.log(2))
        
        return entropy
    
    def attractor_analysis(self, embedded: np.ndarray) -> dict:
        """
        Analyze reconstructed attractor properties.
        
        Parameters:
        -----------
        embedded : np.ndarray
            Embedded time series
            
        Returns:
        --------
        dict
            Dictionary containing attractor metrics
        """
        results = {}
        
        try:
            # Convex hull volume
            if embedded.shape[1] >= 2:
                hull = ConvexHull(embedded)
                results['convex_hull_volume'] = hull.volume
                results['convex_hull_area'] = hull.area if embedded.shape[1] == 2 else np.nan
            else:
                results['convex_hull_volume'] = np.nan
                results['convex_hull_area'] = np.nan
            
            # Correlation dimension estimate (simplified)
            results['correlation_dimension'] = self._estimate_correlation_dimension(embedded)
            
            # Lyapunov exponent estimate (simplified)
            results['lyapunov_estimate'] = self._estimate_lyapunov(embedded)
            
            logger.info("Attractor analysis completed")
            
        except Exception as e:
            logger.error(f"Attractor analysis failed: {e}")
            results = {key: np.nan for key in ['convex_hull_volume', 'convex_hull_area', 
                                             'correlation_dimension', 'lyapunov_estimate']}
        
        return results
    
    def _estimate_correlation_dimension(self, embedded: np.ndarray, 
                                      n_samples: int = 1000) -> float:
        """
        Estimate correlation dimension using the Grassberger-Procaccia algorithm.
        """
        if len(embedded) > n_samples:
            idx = np.random.choice(len(embedded), n_samples, replace=False)
            embedded = embedded[idx]
        
        # Compute pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(embedded)
        
        # Compute correlation sum for different radii
        radii = np.logspace(np.log10(np.min(distances[distances > 0])), 
                           np.log10(np.max(distances)), 20)
        
        correlation_sums = []
        for r in radii:
            correlation_sums.append(np.mean(distances < r))
        
        # Estimate dimension from slope
        log_r = np.log(radii[1:-1])
        log_c = np.log(np.array(correlation_sums)[1:-1])
        
        # Linear fit to estimate slope
        valid_idx = ~(np.isinf(log_c) | np.isnan(log_c))
        if np.sum(valid_idx) > 2:
            slope = np.polyfit(log_r[valid_idx], log_c[valid_idx], 1)[0]
            return max(0, slope)
        
        return np.nan
    
    def _estimate_lyapunov(self, embedded: np.ndarray) -> float:
        """
        Simplified Lyapunov exponent estimation.
        """
        # This is a very simplified implementation
        # In practice, would use more sophisticated methods
        try:
            diffs = np.diff(embedded, axis=0)
            norms = np.linalg.norm(diffs, axis=1)
            norms = norms[norms > 0]
            if len(norms) > 10:
                return np.mean(np.log(norms))
            return np.nan
        except:
            return np.nan


class LorenzSystem:
    """
    Lorenz system generator with configurable parameters.
    """
    
    def __init__(self, sigma: float = 10.0, beta: float = 8/3, rho: float = 28.0):
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
    
    def generate(self, t: np.ndarray, 
                initial_conditions: Tuple[float, float, float] = (0.0, 1.0, 1.05),
                dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate Lorenz system trajectory.
        
        Parameters:
        -----------
        t : np.ndarray
            Time points
        initial_conditions : Tuple[float, float, float]
            Initial conditions (x0, y0, z0)
        dt : float
            Integration time step
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (x, y, z) trajectories
        """
        n = len(t)
        x = np.zeros(n)
        y = np.zeros(n)
        z = np.zeros(n)
        
        x[0], y[0], z[0] = initial_conditions
        
        for i in range(1, n):
            dx = self.sigma * (y[i-1] - x[i-1])
            dy = x[i-1] * (self.rho - z[i-1]) - y[i-1]
            dz = x[i-1] * y[i-1] - self.beta * z[i-1]
            
            x[i] = x[i-1] + dx * dt
            y[i] = y[i-1] + dy * dt
            z[i] = z[i-1] + dz * dt
        
        return x, y, z


def comprehensive_analysis_demo():
    """
    Demonstration of comprehensive nonlinear time series analysis.
    """
    # Initialize analyzer
    analyzer = NonlinearTimeSeriesAnalyzer(normalize_data=True)
    
    # Generate Lorenz system data
    lorenz = LorenzSystem()
    t = np.linspace(0, 50, 5000)
    x, y, z = lorenz.generate(t)
    
    # Use x component for analysis
    data = x
    
    print("=== Nonlinear Time Series Analysis ===")
    print(f"Data length: {len(data)}")
    print(f"Data range: [{np.min(data):.3f}, {np.max(data):.3f}]")
    
    # Step 1: Find optimal embedding parameters
    optimal_delay, optimal_dim = analyzer.optimal_embedding_parameters(data)
    
    # Step 2: Perform embedding
    embedded = analyzer.time_delay_embedding(data, delay=optimal_delay, dimension=optimal_dim)
    
    # Step 3: Wavelet analysis
    wavelet_power = analyzer.wavelet_spectrum(data)
    
    # Step 4: Transfer entropy analysis
    # Use x -> y causality
    te_xy = analyzer.transfer_entropy(x[:-1], y[1:])  # x influences y
    te_yx = analyzer.transfer_entropy(y[:-1], x[1:])  # y influences x
    
    # Step 5: Attractor analysis
    attractor_metrics = analyzer.attractor_analysis(embedded)
    
    # Results summary
    print("\n=== Analysis Results ===")
    print(f"Optimal embedding delay: {optimal_delay}")
    print(f"Optimal embedding dimension: {optimal_dim}")
    print(f"Transfer entropy X->Y: {te_xy:.6f}")
    print(f"Transfer entropy Y->X: {te_yx:.6f}")
    print(f"Convex hull volume: {attractor_metrics['convex_hull_volume']:.6f}")
    print(f"Correlation dimension: {attractor_metrics['correlation_dimension']:.6f}")
    print(f"Lyapunov estimate: {attractor_metrics['lyapunov_estimate']:.6f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Original time series
    axes[0, 0].plot(t[:1000], data[:1000], 'b-', linewidth=0.8)
    axes[0, 0].set_title('Original Time Series')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Phase space embedding
    axes[0, 1].scatter(embedded[:, 0], embedded[:, 1], s=0.5, alpha=0.6, c='red')
    axes[0, 1].set_title(f'Phase Space (delay={optimal_delay})')
    axes[0, 1].set_xlabel('x(t)')
    axes[0, 1].set_ylabel('x(t+τ)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 3D attractor (if dimension >= 3)
    if embedded.shape[1] >= 3:
        ax_3d = fig.add_subplot(2, 3, 3, projection='3d')
        ax_3d.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], 
                     s=0.3, alpha=0.6, c='green')
        ax_3d.set_title('3D Attractor')
        ax_3d.set_xlabel('x(t)')
        ax_3d.set_ylabel('x(t+τ)')
        ax_3d.set_zlabel('x(t+2τ)')
        axes[0, 2].remove()  # Remove the 2D subplot
    
    # 4. Wavelet power spectrum
    im = axes[1, 0].imshow(wavelet_power, extent=[0, len(data), 1, wavelet_power.shape[0]], 
                          aspect='auto', cmap='viridis', origin='lower')
    axes[1, 0].set_title('Wavelet Power Spectrum')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Scale')
    plt.colorbar(im, ax=axes[1, 0], label='Power')
    
    # 5. Transfer entropy comparison
    te_values = [te_xy, te_yx]
    te_labels = ['X→Y', 'Y→X']
    bars = axes[1, 1].bar(te_labels, te_values, color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title('Transfer Entropy')
    axes[1, 1].set_ylabel('TE Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, te_values):
        if not np.isnan(value):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom')
    
    # 6. Attractor metrics summary
    axes[1, 2].axis('off')
    metrics_text = f"""Attractor Analysis:
    
    Convex Hull Volume: {attractor_metrics['convex_hull_volume']:.4f}
    
    Correlation Dimension: {attractor_metrics['correlation_dimension']:.4f}
    
    Lyapunov Estimate: {attractor_metrics['lyapunov_estimate']:.4f}
    
    Embedding Parameters:
    • Delay: {optimal_delay}
    • Dimension: {optimal_dim}"""
    
    axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run comprehensive analysis
    comprehensive_analysis_demo()
