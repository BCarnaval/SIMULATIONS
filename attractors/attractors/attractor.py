#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""This module defines the 'Attractor' objects, their attributes and methods.
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import solvers as solvers
import useful_funcs as ufs


class ChaosAttractor:
    """System objects definition.

    Attributes
    ----------
    name (self.name): str, default=None
        Name given by user to differentiate system & objects.

    init_state (self.r): np.ndarray, shape=(3, ), default=None
        Initial position of the simulation.

    time_domain (self.ts): np.ndarray, shape=(n, ), default=None
        Time domain of the simulation.

    jacobian: np.ndarray, shape=(3, 3), default=None
        Jacobian matrix of the differential system.

    trajectory: np.ndarray, shape=(n, 3), default=None
        Position at each moment of 'self.ts' of the simulation.
    """

    def __init__(self, name: str, init_state: np.ndarray,
                 time_domain: np.ndarray) -> None:
        """Associating class attributes to given constants.
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
        """Accessing differential system of an object by using its private
        method.
        """
        raise NotImplementedError(
            "Subclasses must implement 'differential_system' mehtod.")

    def get_jacobian_matrix(self, position: np.ndarray) -> np.ndarray:
        """Accessing the jacobian matrix of the differential system of an
        object by using its private method.
        """
        raise NotImplementedError(
            "Subclasses must implement 'get_jacobian_matrix' mehtod.")

    def compute_trajectory(self, state0: np.ndarray) -> np.ndarray:
        """Using 'Solver' objects to compute the full trajectory inside
        studied attractor.

        Parameters
        ----------
        state0: np.ndarray, shape=(3, ), default=None
            Initial position for the simulation.

        Returns
        -------
        _: np.ndarray, shape=(self.n, 3)
            All positions for a given time domain.
        """
        solver = solvers.Solver(
            func=self.differential_system,
            ts=self.ts,
            y0=state0,
        )

        return solver.scipy_solver()

    def plot_trajectory(self, save: bool = False) -> plt.figure:
        """Plots the computed trajectory using 'matplotlib'.

        Parameters
        ----------
        save: bool, default=False
            If set to True, saves the outputed figure in predetermined
            location using attractor's name attribute.
        """
        fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))

        cmap = plt.cm.GnBu
        positions, n, s = self.trajectory, self.n, 25
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

        axs.plot(x[0], y[0], z[0], label=f'Trajectory: {self.name} attractor')
        for i in range(0, n - s, s):
            axs.plot(x[i: i + s + 1],
                     y[i: i + s + 1],
                     z[i: i + s + 1],
                     color=cmap(i / n)
                     )

        handles, labels = axs.get_legend_handles_labels()
        axs.set_xlabel(r'$X$')
        axs.set_ylabel(r'$Y$')
        axs.set_zlabel(r'$Z$')
        axs.legend(handles=handles, labels=labels)

        if save:
            plt.savefig(f'./attractors/figs/trajectories/traj_{self.name}.png')

        plt.show()

        return

    def animate_trajectory(self, show: bool = True) -> None:
        """Animates computed trajectory using 'matploltib.animation' module.

        Parameters
        ----------
        show: bool, default=True
            If set to False, it saves the animation as an *.mp4 instead of
            displaying it directly in a new window.
        """
        traj = self.trajectory
        fig, axs = plt.subplots(subplot_kw=dict(projection='3d'))
        line, = axs.plot([], [], [], lw=0.75,
                         label=f'Trajectory: {self.name} attractor')
        particle, = axs.plot([], [], [], 'o', color='C5')

        # Set x labels & limits
        axs.set_xlim3d([min(traj[:, 0]), max(traj[:, 0])])
        axs.set_xlabel(r'$X$')

        # Set y labels & limits
        axs.set_ylim3d([min(traj[:, 1]), max(traj[:, 1])])
        axs.set_ylabel(r'$Y$')

        # Set z labels & limits
        axs.set_zlim3d([min(traj[:, 2]), max(traj[:, 2])])
        axs.set_zlabel(r'$Z$')

        plt.legend()

        def animate(i):
            """Compute steps of the animation.
            """
            # Update line trajectory
            line.set_data(traj[:i, :2].T)
            line.set_3d_properties(traj[:i, 2])

            # Update particle trajectory
            particle.set_data(traj[i, :2].T)
            particle.set_3d_properties(traj[i, 2])
            return line, particle,

        interval = 1000 * (self.ts[1] - self.ts[0])
        ani = animation.FuncAnimation(
            fig, animate, interval=interval, frames=range(self.n), blit=True)

        if show:
            plt.show()
        else:
            ani.save(
                f'./attractors/animations/{self.name}.mp4', dpi=300, fps=90)

        return

    def compute_lyapunovs(self) -> tuple[np.ndarray]:
        """Docs
        """
        def diff_system(state: np.ndarray):
            """Docs
            """
            diff = self.differential_system(state, self.ts)
            J = self.get_jacobian_matrix(state)
            return diff, J

        def LES(state: np.ndarray):
            """Docs
            """
            U = state[3:12].reshape([3, 3])

            f, df = diff_system(state[:3])
            A = U.T.dot(df.dot(U))
            dl = np.diag(A).copy()

            for i in range(3):
                A[i, i] = 0
                for j in range(i + 1, 3):
                    A[i, j] = -A[j, i]

            dU = U.dot(A)

            return np.concatenate([f, dU.flatten(), dl])

        # Initializing positon, tangent vector and Lyapunov spectrum
        r = self.r
        U = np.eye(3)
        L = np.zeros(3)

        # Solving system using scipy odeint
        state0 = np.concatenate([r, U.flatten(), L])
        sols = sp.integrate.odeint(lambda A, t: LES(A), state0, self.ts)

        # Retreive Lyapunov exponents from concatenated array
        lyaps = (sols[50:, 12:15].T / self.ts[50:]).T
        l_1, l_2, l_3 = lyaps[:, 0], lyaps[:, 1], lyaps[:, 2]

        return l_1, l_2, l_3

    def plot_lyapunovs(self, save: bool = False) -> plt.figure:
        """Plots the Lyapunov spectrum over time in the simulation.

        Parameters
        ----------
        save: bool, default=False
            If set to True, saves the outputed figure in predetermined
            location using attractor's name attribute.
        """
        # Global plot attributes
        colors = ['C0', 'C1', 'C2']
        labels = ['$L_1$', '$L_2$', '$L_3$']
        lyaps = self.compute_lyapunovs()
        extraps = [ufs.epsilon(lyapunov) for lyapunov in lyaps]

        # Using a loop to plot whole spectrum
        for lyap, label, color, extra in zip(lyaps, labels, colors, extraps):
            plt.plot(self.ts[50:], lyap, label=label +
                     r'$\simeq{:.2f}$'.format(extra))
            plt.hlines(xmin=self.ts[0], xmax=self.ts[-1], y=ufs.epsilon(
                lyap), linestyle='dashdot', lw=0.5, color=color)

        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel(r'Lyapunov spectrum ($L_i$)')

        if save:
            plt.savefig(f'./attractors/figs/lyapunovs/lyap_{self.name}.png')

        plt.show()

        return


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
