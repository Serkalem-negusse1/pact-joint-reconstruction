import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Callable
from dataclasses import dataclass


@dataclass
class ADMMParameters:
    rho: float = 1.0
    max_iter: int = 1000
    abs_tol: float = 1e-4
    rel_tol: float = 1e-2
    adaptive_rho: bool = True


class ADMMSolver:
    """
    Alternating Direction Method of Multipliers solver
    for the joint reconstruction problem
    """
    
    def __init__(self, params: ADMMParameters = ADMMParameters()):
        self.params = params
    
    def solve(self, data_fidelity: Callable,
              prox_x: Callable,
              prox_z: Callable,
              x0: jnp.ndarray,
              D: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """
        Solve optimization problem using ADMM
        
        min f(x) + g(z) s.t. Dx = z
        """
        # Initialize
        x = x0
        z = D @ x
        u = jnp.zeros_like(z)
        rho = self.params.rho
        
        # Storage
        history = {
            'primal_res': [],
            'dual_res': [],
            'objective': []
        }
        
        for k in range(self.params.max_iter):
            # x-update: min f(x) + (ρ/2)‖Dx - z + u‖²
            def x_subproblem(x):
                return data_fidelity(x) + (rho/2) * jnp.sum((D @ x - z + u)**2)
            
            x = prox_x(x, x_subproblem)
            
            # z-update: min g(z) + (ρ/2)‖Dx - z + u‖²
            z_old = z
            z = prox_z(D @ x + u)
            
            # u-update
            u = u + D @ x - z
            
            # Compute residuals
            primal_res = jnp.linalg.norm(D @ x - z)
            dual_res = rho * jnp.linalg.norm(D.T @ (z - z_old))
            
            # Compute objective
            obj = data_fidelity(x)  # + g(z) term
            
            # Store history
            history['primal_res'].append(float(primal_res))
            history['dual_res'].append(float(dual_res))
            history['objective'].append(float(obj))
            
            # Check convergence
            eps_primal = (np.sqrt(z.size) * self.params.abs_tol + 
                         self.params.rel_tol * max(jnp.linalg.norm(D @ x), 
                                                  jnp.linalg.norm(z)))
            eps_dual = (np.sqrt(x.size) * self.params.abs_tol + 
                       self.params.rel_tol * jnp.linalg.norm(D.T @ u))
            
            if primal_res < eps_primal and dual_res < eps_dual:
                break
            
            # Adaptive rho update
            if self.params.adaptive_rho and k % 10 == 0:
                if primal_res > 10 * dual_res:
                    rho *= 2
                    u /= 2
                elif dual_res > 10 * primal_res:
                    rho /= 2
                    u *= 2
        
        return x, history