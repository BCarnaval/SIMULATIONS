#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Bouali attractor objects definition.
"""
import numpy as np

from attractor import ChaosAttractor


class Bouali(ChaosAttractor):
    """Docs
    """

    def __init__(self, name: str, init_state: np.ndarray,
                 time_domain: np.ndarray) -> None:
        """Docs
        """
        self.alpha, self.beta, self.gamma, self.mu = 3.0, 2.2, 1, 1.5e-3
        super().__init__(name, init_state, time_domain)

    def __len__(self) -> int:
        return self.ts.shape[0]

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Docs
        """
        r = position
        jacobian = np.array([
            [self.alpha * (1 - r[1]), -self.alpha * r[0], -self.beta],
            [2 * self.gamma * r[0] * r[1], -self.gamma * (1 - r[0]**2), 0],
            [self.mu, 0, 0]
        ])
        return jacobian

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Docs
        """
        # Initial position
        x, y, z = r

        # Update coordinates
        x_dot = self.alpha * x * (1 - y) - self.beta * z
        y_dot = -self.gamma * y * (1 - x**2)
        z_dot = self.mu * x

        return np.array([x_dot, y_dot, z_dot])


if __name__ == '__main__':
    state0 = np.array([0.2, 0.2, -0.1])
    time = np.linspace(0, 1000, 100000)

    ex = Bouali(name='bouali',
                init_state=state0,
                time_domain=time
                )

    lyaps = ex.compute_lyapunovs()
    print(lyaps)
