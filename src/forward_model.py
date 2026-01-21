import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import scipy.sparse as sp


class AcousticForwardModel:
    """
    Acoustic forward model for photoacoustic computed tomography
    Implements the k-space pseudospectral method
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (256, 256),
                 dx: float = 0.32e-3,  # meters
                 c0: float = 1540.0,  # m/s
                 dt: float = 64e-9,  # seconds
                 n_sensors: int = 128):
        
        self.grid_size = grid_size
        self.nx, self.ny = grid_size
        self.dx = dx
        self.c0 = c0
        self.dt = dt
        
        # Create k-space operators
        self._setup_kspace_operators()
        
        # Create sensor array
        self.sensor_positions = self._create_sensor_array(n_sensors)
        
    def _setup_kspace_operators(self):
        """Setup k-space differentiation operators"""
        # Spatial frequency grids
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, self.dx)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        
        # k-space operator for perfect matched layer
        k = np.sqrt(KX**2 + KY**2)
        k_max = np.pi / self.dx
        self.kappa = np.sinc(k / (2 * k_max))
        
        # Store for JAX
        self.kappa_jax = jnp.array(self.kappa)
        
    def _create_sensor_array(self, n_sensors: int):
        """Create circular sensor array"""
        radius = 0.072  # 72mm radius
        angles = np.linspace(0, 2*np.pi, n_sensors, endpoint=False)
        sensors = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles)
        ])
        return sensors
    
    @jax.jit
    def forward_step(self, p: jnp.ndarray, ux: jnp.ndarray, uy: jnp.ndarray,
                    c: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Single forward time step using k-space method
        
        Args:
            p: Pressure field
            ux, uy: Particle velocity components
            c: Speed of sound distribution
            
        Returns:
            Updated p, ux, uy
        """
        # Compute gradients in Fourier domain
        p_hat = jnp.fft.fft2(p)
        kx = jnp.fft.fftfreq(self.nx, self.dx) * 2 * np.pi
        ky = jnp.fft.fftfreq(self.ny, self.dx) * 2 * np.pi
        
        # Update velocity
        grad_p_x = jnp.real(jnp.fft.ifft2(1j * kx[:, None] * p_hat))
        grad_p_y = jnp.real(jnp.fft.ifft2(1j * ky[None, :] * p_hat))
        
        ux_new = ux - self.dt * grad_p_x / 1000  # assuming constant density
        uy_new = uy - self.dt * grad_p_y / 1000
        
        # Update pressure
        div_u = jnp.real(jnp.fft.ifft2(
            1j * kx[:, None] * jnp.fft.fft2(ux_new) +
            1j * ky[None, :] * jnp.fft.fft2(uy_new)
        ))
        
        p_new = p - self.dt * (c**2) * 1000 * div_u  # c² * rho * div(u)
        
        # Apply k-space operator
        p_hat_new = jnp.fft.fft2(p_new)
        p_new = jnp.real(jnp.fft.ifft2(p_hat_new * self.kappa_jax))
        
        return p_new, ux_new, uy_new
    
    def forward_solve(self, p0: np.ndarray, c: np.ndarray, 
                     n_steps: int = 1750) -> np.ndarray:
        """
        Full forward simulation
        
        Args:
            p0: Initial pressure distribution
            c: Speed of sound distribution
            n_steps: Number of time steps
            
        Returns:
            Measurement data at sensor locations
        """
        # Initialize fields
        p = p0.copy().astype(np.float32)
        ux = np.zeros_like(p)
        uy = np.zeros_like(p)
        
        # Convert to JAX arrays
        p_jax = jnp.array(p)
        ux_jax = jnp.array(ux)
        uy_jax = jnp.array(uy)
        c_jax = jnp.array(c)
        
        # Storage for sensor measurements
        measurements = np.zeros((n_steps, len(self.sensor_positions)))
        
        # Time stepping
        for t in range(n_steps):
            p_jax, ux_jax, uy_jax = self.forward_step(p_jax, ux_jax, uy_jax, c_jax)
            
            # Sample at sensor positions (simplified)
            if t % 10 == 0:  # Subsample for efficiency
                p_np = np.array(p_jax)
                for i, (x, y) in enumerate(self.sensor_positions):
                    # Simple nearest neighbor sampling
                    xi = int((x / self.dx) + self.nx // 2)
                    yi = int((y / self.dx) + self.ny // 2)
                    if 0 <= xi < self.nx and 0 <= yi < self.ny:
                        measurements[t, i] = p_np[xi, yi]
        
        return measurements