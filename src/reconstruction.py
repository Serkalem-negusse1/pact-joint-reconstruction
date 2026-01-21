import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Tuple, Optional
from dataclasses import dataclass
from .constraints import Constraint


@dataclass
class ReconstructionResult:
    """Container for reconstruction results"""
    p0_est: np.ndarray
    c_est: np.ndarray
    loss_history: List[float]
    iterations: int
    convergence: bool


class JointReconstructor:
    """
    Joint reconstruction of initial pressure and speed of sound
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (256, 256),
                 dx: float = 0.32e-3, c0: float = 1540.0):
        
        self.grid_size = grid_size
        self.nx, self.ny = grid_size
        self.dx = dx
        self.c0 = c0
        
        self.constraints: List[Constraint] = []
        self.loss_history = []
        
    def add_constraint(self, constraint: Constraint):
        """Add a constraint to the reconstruction"""
        self.constraints.append(constraint)
    
    def data_fidelity(self, p0_est: jnp.ndarray, c_est: jnp.ndarray,
                     measurements: jnp.ndarray, forward_model) -> float:
        """
        Compute data fidelity term: ½‖M H(c) p0 - y‖²
        """
        # Simulate measurements
        y_sim = forward_model.forward_solve(np.array(p0_est), np.array(c_est))
        
        # Compute residual
        residual = y_sim - measurements
        
        # Return squared norm
        return 0.5 * jnp.sum(residual**2)
    
    @jax.jit
    def gradient_p0(self, p0: jnp.ndarray, c: jnp.ndarray,
                   measurements: jnp.ndarray, forward_model) -> jnp.ndarray:
        """
        Compute gradient with respect to p0 using automatic differentiation
        """
        def loss_fn(p0):
            return self.data_fidelity(p0, c, measurements, forward_model)
        
        return jax.grad(loss_fn)(p0)
    
    @jax.jit
    def gradient_c(self, p0: jnp.ndarray, c: jnp.ndarray,
                  measurements: jnp.ndarray, forward_model) -> jnp.ndarray:
        """
        Compute gradient with respect to c using automatic differentiation
        """
        def loss_fn(c):
            return self.data_fidelity(p0, c, measurements, forward_model)
        
        return jax.grad(loss_fn)(c)
    
    def reconstruct(self, measurements: np.ndarray,
                   n_iter: int = 100,
                   learning_rate: float = 1e-3,
                   verbose: bool = True) -> ReconstructionResult:
        """
        Perform joint reconstruction using alternating minimization
        
        Args:
            measurements: Sensor measurements
            n_iter: Number of iterations
            learning_rate: Step size for gradient descent
            verbose: Print progress
            
        Returns:
            ReconstructionResult object
        """
        # Initialize estimates
        p0_est = np.zeros(self.grid_size, dtype=np.float32)
        c_est = np.ones(self.grid_size, dtype=np.float32) * self.c0
        
        # Create forward model
        forward_model = AcousticForwardModel(
            grid_size=self.grid_size,
            dx=self.dx,
            c0=self.c0
        )
        
        # Convert to JAX arrays
        p0_est_jax = jnp.array(p0_est)
        c_est_jax = jnp.array(c_est)
        measurements_jax = jnp.array(measurements)
        
        # Alternating minimization
        for iteration in range(n_iter):
            # Update p0
            grad_p0 = self.gradient_p0(p0_est_jax, c_est_jax, 
                                      measurements_jax, forward_model)
            p0_est_jax = p0_est_jax - learning_rate * grad_p0
            
            # Apply constraints to p0
            for constraint in self.constraints:
                p0_est_jax = constraint.apply(p0_est_jax, 'p0')
            
            # Update c
            grad_c = self.gradient_c(p0_est_jax, c_est_jax,
                                    measurements_jax, forward_model)
            c_est_jax = c_est_jax - learning_rate * grad_c
            
            # Apply constraints to c
            for constraint in self.constraints:
                c_est_jax = constraint.apply(c_est_jax, 'c')
            
            # Compute loss
            loss = self.data_fidelity(p0_est_jax, c_est_jax,
                                     measurements_jax, forward_model)
            self.loss_history.append(float(loss))
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6e}")
        
        # Convert back to numpy
        p0_est_result = np.array(p0_est_jax)
        c_est_result = np.array(c_est_jax)
        
        return ReconstructionResult(
            p0_est=p0_est_result,
            c_est=c_est_result,
            loss_history=self.loss_history,
            iterations=n_iter,
            convergence=len(self.loss_history) > 0 and 
                       self.loss_history[-1] < 1e-3
        )