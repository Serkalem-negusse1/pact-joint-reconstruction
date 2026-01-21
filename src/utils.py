import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def create_circular_sensor_array(radius: float = 0.072,
                                n_sensors: int = 128) -> np.ndarray:
    """Create circular array of sensors"""
    angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
    sensors = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    return sensors


def generate_breast_phantom(grid_size: Tuple[int, int] = (256, 256),
                           breast_type: str = 'D',
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate simplified breast phantom for testing
    
    Args:
        grid_size: Size of the phantom
        breast_type: A, B, C, or D type breast
        seed: Random seed for reproducibility
    
    Returns:
        p0: Initial pressure distribution
        c: Speed of sound distribution
    """
    np.random.seed(seed)
    nx, ny = grid_size
    
    # Create coordinate grid
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    
    # Base breast shape (ellipse)
    theta = 0.3  # Rotation angle
    a, b = 0.6, 0.8  # Semi-axes
    
    # Rotated ellipse equation
    breast_mask = ((X*np.cos(theta) + Y*np.sin(theta))**2 / a**2 +
                  (X*np.sin(theta) - Y*np.cos(theta))**2 / b**2) <= 1
    
    # Add some internal structures (vessels, glands)
    structures = np.zeros_like(R)
    n_structures = {'A': 3, 'B': 8, 'C': 15, 'D': 25}[breast_type]
    
    for _ in range(n_structures):
        cx, cy = np.random.uniform(-0.5, 0.5, 2)
        r = np.random.uniform(0.05, 0.15)
        intensity = np.random.uniform(0.3, 1.0)
        
        # Gaussian structure
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        structures += intensity * np.exp(-dist**2 / (2*r**2))
    
    # Initial pressure (optical absorption)
    p0 = structures * breast_mask
    
    # Speed of sound distribution
    # Base SOS in water: 1540 m/s
    # Fat: ~1450 m/s, Glandular tissue: ~1520-1620 m/s
    c_base = 1540.0  # m/s
    
    # Adjust based on breast type
    sos_factors = {'A': 0.94, 'B': 0.96, 'C': 0.98, 'D': 1.0}
    factor = sos_factors[breast_type]
    
    # Add heterogeneity
    c_variation = 0.1 * structures  # ±10% variation
    c = c_base * (factor + c_variation) * breast_mask + c_base * (~breast_mask)
    
    # Normalize p0 for visualization
    p0 = (p0 - p0.min()) / (p0.max() - p0.min() + 1e-8)
    
    return p0, c


def plot_reconstruction_results(p0_true, c_true, p0_est, c_est,
                               loss_history=None, figsize=(15, 10)):
    """Plot reconstruction results"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # True distributions
    im1 = axes[0, 0].imshow(p0_true.T, cmap='hot', origin='lower')
    axes[0, 0].set_title('True Initial Pressure')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(c_true.T, cmap='viridis', origin='lower')
    axes[0, 1].set_title('True Speed of Sound')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Estimated distributions
    im3 = axes[1, 0].imshow(p0_est.T, cmap='hot', origin='lower')
    axes[1, 0].set_title('Estimated Initial Pressure')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(c_est.T, cmap='viridis', origin='lower')
    axes[1, 1].set_title('Estimated Speed of Sound')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Error maps
    p0_error = np.abs(p0_true - p0_est)
    im5 = axes[0, 2].imshow(p0_error.T, cmap='Reds', origin='lower')
    axes[0, 2].set_title('Pressure Error')
    plt.colorbar(im5, ax=axes[0, 2])
    
    c_error = np.abs(c_true - c_est)
    im6 = axes[1, 2].imshow(c_error.T, cmap='Reds', origin='lower')
    axes[1, 2].set_title('SOS Error')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Loss history if provided
    if loss_history is not None:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Convergence History')
        ax.grid(True)
        ax.set_yscale('log')
    
    plt.tight_layout()
    return fig