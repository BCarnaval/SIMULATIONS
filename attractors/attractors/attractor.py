#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""This module defines the 'Attractor' objects, their attributes and methods.
"""
import numpy as np
import matplotlib.pyplot as plt

from solvers import Solver


class ChaosAttractor:
    """System objects definition.

    Attributes
    ----------
    """

    def __init__(self, name: str, init_state: np.ndarray,
                 time_domain: np.ndarray) -> None:
        """Docs
        """
        self.name = name
        self.r = init_state
        self.ts = time_domain

        self.n = self.__len__()
        self.jacobian = self.get_jacobian_matrix(position=init_state)
        self.trajectory = self.compute_trajectory(state0=init_state)

    def __len__(self) -> int:
        return len(self.ts)

    def differential_system(self, r: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Docs
        """
        raise NotImplementedError(
            "Subclasses must implement 'differential_system' mehtod.")

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Docs
        """
        raise NotImplementedError(
            "Subclasses must implement 'get_jacobian_matrix' mehtod.")

    def compute_trajectory(self, state0: np.ndarray) -> np.ndarray:
        """Docs
        """
        solver = Solver(
            func=self.differential_system,
            ts=self.ts,
            y0=state0,
        )

        return solver.scipy_solver()

    def plot_trajectory(self) -> plt.figure:
        """Docs
        """
        fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))

        cmap = plt.cm.GnBu
        positions, n, s = self.trajectory, self.__len__(), 25
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        for i in range(0, n - s, s):
            axs.plot(x[i: i + s + 1],
                     y[i: i + s + 1],
                     z[i: i + s + 1],
                     color=cmap(i / n)
                     )

        plt.show()

        return

    def compute_lyapunovs(self):
        """Docs
        """
        # Initializing tangent vector
        U = np.eye(3)

        # Init. Lyapunov exponents array
        lyaps = np.zeros(shape=(self.n, 3))

        dt = self.ts[1] - self.ts[0]
        trajectory = self.trajectory
        for idx, state in enumerate(trajectory):
            # Get Jacobian matrix foreach state
            J = self.get_jacobian_matrix(state)
            U_n = (np.eye(3) + J * dt) @ U

            # Get Q, R decomposition matrices
            Q, R = np.linalg.qr(U_n)

            lyaps[idx] = np.log(np.abs(np.diag(R))) / self.ts[-1]

            U = Q

        plt.plot(self.ts, lyaps)
        plt.show()

        return lyaps.sum(axis=0)


class Simulation:
    """Attractor objects definition.

    Attributes
    ----------
    """

    def __init__(self, name: str, system: ChaosAttractor) -> None:
        """Docs
        """
        self.name = name
        self.system = system

    def __repr__(self) -> None:
        """Docs
        """
        return str(" ".join(["Attractor", self.name, "stored."]))


if __name__ == "__main__":
    pass
