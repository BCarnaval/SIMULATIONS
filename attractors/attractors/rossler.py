#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Rossler attractor objects definition.
"""
import numpy as np

from attractor import ChaosAttractor


class Rossler(ChaosAttractor):
    """Docs
    """

    def __init__(self, name: str, init_state: np.ndarray,
                 time_domain: np.ndarray) -> None:
        """Docs
        """
        self.alpha, self.beta, self.gamma = 0.2, 0.2, 5.7
        super().__init__(name, init_state, time_domain)

    def __len__(self) -> int:
        return self.ts.shape[0]

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Docs
        """
        r = position
        jacobian = np.array([
            [0, -1, -1],
            [1, self.alpha, 0],
            [r[2], 0, r[0] - self.gamma]
        ])
        return jacobian

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Docs
        """
        # Initial position
        x, y, z = r

        # Update coordinates
        x_dot = -y - z
        y_dot = x + self.alpha * y
        z_dot = self.beta + z * (x - self.gamma)

        return np.array([x_dot, y_dot, z_dot])


if __name__ == '__main__':
    state0 = np.array([0, 1, -1])
    time = np.linspace(0, 1000, 100000)

    ex = Rossler(name='rossler',
                 init_state=state0,
                 time_domain=time
                 )

    lyaps = ex.compute_lyapunovs()
    print(lyaps)
