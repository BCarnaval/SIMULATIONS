#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Lorenz attractor objects definition.
"""
import numpy as np

from attractor import ChaosAttractor


class Lorenz(ChaosAttractor):
    """Docs
    """

    def __init__(self, name, init_state, time_domain) -> None:
        """Docs
        """
        self.sigma, self.rho, self.beta = 10, 28, 8/3
        super().__init__(name, init_state, time_domain)

    def __len__(self) -> int:
        return self.ts.shape[0]

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Docs
        """
        x, y, z = position
        jacobian = np.array([
            [-self.sigma, self.sigma, 0],
            [self.rho - z, -1, -x],
            [y, x, -self.beta]
        ])
        return jacobian

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Docs
        """
        # Initial position
        x, y, z = r

        # Update coordinates
        x_dot = self.sigma * (y - x)
        y_dot = self.rho * x - y - x * z
        z_dot = -self.beta * z + x * y

        return np.array([x_dot, y_dot, z_dot])


if __name__ == '__main__':
    state0 = np.array([1, 1, 1])
    time = np.linspace(0, 100, 100000)

    ex = Lorenz(name='lorenz',
                init_state=state0,
                time_domain=time
                )

    lyaps = ex.compute_lyapunovs()
    print(lyaps)
