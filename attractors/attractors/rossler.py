#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Rossler attractor objects definition.
"""
import numpy as np

from attractor import ChaosAttractor


class Rossler(ChaosAttractor):
    """Rossler attractors specific attributes and methods.

    Attributes
    ----------
    name: str, default='rossler'
        Name of the attractor given by user.

    init_state: np.ndarray, shape=(3, ), default=np.array([1, 1, -1])
        Initial position of the particle.

    time_domain: np.ndarray, shape=(n, ), default=np.linspace(0, 500, 10000)
        Time domain of the simulation.
    """

    def __init__(self, name: str = 'rossler',
                 init_state: np.ndarray = np.array([1, 1, -1]),
                 time_domain: np.ndarray = np.linspace(0, 500, 10000)) -> None:
        """Associating class attributes and setup inheritance with
        'ChaosAttractor' objects.
        """
        self.alpha, self.beta, self.gamma = 0.2, 0.2, 5.7
        super().__init__(name, init_state, time_domain)

    def __len__(self) -> int:
        return self.ts.shape[0]

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Compute jacobian matrix of the system at a specific position of the
        trajectory.

        Parameters
        ----------
        position: np.ndarray, shape=(3, ), default=None
            Position at which compute the jacobian.

        Returns
        -------
        jacobian: np.ndarray, shape=(3, 3)
            Jacobian matrix evaluated at specific position.
        """
        r = position
        jacobian = np.array([
            [0, -1, -1],
            [1, self.alpha, 0],
            [r[2], 0, r[0] - self.gamma]
        ])
        return jacobian

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Defines the differential equations system of Rossler attractor.
        """
        # Initial position
        x, y, z = r

        # Update coordinates
        x_dot = -y - z
        y_dot = x + self.alpha * y
        z_dot = self.beta + z * (x - self.gamma)

        return np.array([x_dot, y_dot, z_dot])


if __name__ == '__main__':
    ex = Rossler()
