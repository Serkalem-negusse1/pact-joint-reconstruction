import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Union
from abc import ABC, abstractmethod


class Constraint(ABC):
    """Base class for reconstruction constraints"""
    
    @abstractmethod
    def apply(self, x: jnp.ndarray, field_type: str = 'p0') -> jnp.ndarray:
        """Apply constraint to the field"""
        pass


class SupportConstraint(Constraint):
    """Support constraint - enforce known region"""
    
    def __init__(self, support_mask: np.ndarray, background_value: float = 0.0):
        self.support_mask = jnp.array(support_mask.astype(bool))
        self.background_value = background_value
    
    def apply(self, x: jnp.ndarray, field_type: str = 'p0') -> jnp.ndarray:
        # Set values outside support to background value
        return jnp.where(self.support_mask, x, self.background_value)


class BoundConstraint(Constraint):
    """Bound constraint - enforce min/max values"""
    
    def __init__(self, lower_bound: float, upper_bound: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def apply(self, x: jnp.ndarray, field_type: str = 'p0') -> jnp.ndarray:
        return jnp.clip(x, self.lower_bound, self.upper_bound)


class TVConstraint(Constraint):
    """Total Variation constraint"""
    
    def __init__(self, tau: float = 1.0, mu: float = 1.0):
        self.tau = tau  # TV bound
        self.mu = mu    # Weight for balancing
    
    def _tv_norm(self, x: jnp.ndarray) -> float:
        """Compute TV norm"""
        dx = jnp.roll(x, -1, axis=0) - x
        dy = jnp.roll(x, -1, axis=1) - x
        return jnp.sum(jnp.sqrt(dx**2 + dy**2))
    
    def apply(self, x: jnp.ndarray, field_type: str = 'p0') -> jnp.ndarray:
        # Simplified TV projection (for demonstration)
        # In practice, this would use ADMM or similar
        current_tv = self._tv_norm(x)
        
        if current_tv > self.tau:
            # Scale down to meet TV constraint
            scale_factor = self.tau / current_tv
            return x * scale_factor
        return x