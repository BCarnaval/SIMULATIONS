#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Bouali attractor objects definition.
"""
import numpy as np

from attractor import ChaosAttractor


class Bouali(ChaosAttractor):
    """Bouali attractors specific attributes and methods.

    Attributes
    ----------
    name: str, default='bouali'
        Name of the attractor given by user.

    init_state: np.ndarray, shape=(3, ), default=np.array([0.2, 0.2, -0.1])
        Initial position of the particle.

    time_domain: np.ndarray, shape=(n, ), default=np.linsapce(0, 500, 10000)
        Time domain of the simulation.
    """

    def __init__(self, name: str = 'bouali',
                 init_state: np.ndarray = np.array([0.2, 0.2, -0.1]),
                 time_domain: np.ndarray = np.linspace(0, 500, 10000)) -> None:
        """Associating class attributes and setup inheritance with
        'ChaosAttractor' objects.
        """
        self.alpha, self.beta, self.gamma, self.mu = 3.0, 2.2, 1, 1.5e-3
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
            [self.alpha * (1 - r[1]), -self.alpha * r[0], -self.beta],
            [2 * self.gamma * r[0] * r[1], -self.gamma * (1 - r[0]**2), 0],
            [self.mu, 0, 0]
        ])
        return jacobian

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Defines the differential equations system of Bouali attractor.
        """
        # Initial position
        x, y, z = r

        # Update coordinates
        x_dot = self.alpha * x * (1 - y) - self.beta * z
        y_dot = -self.gamma * y * (1 - x**2)
        z_dot = self.mu * x

        return np.array([x_dot, y_dot, z_dot])


if __name__ == '__main__':
    ex = Bouali()
