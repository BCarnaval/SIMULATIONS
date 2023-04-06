#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Implementation of 4 differential equations solvers (Euler,
predictor-corrector, Runge-Kutta d'ordre (2 and 4) and scipy.optimize.odeint)
"""

import numpy as np
import scipy as sp
from typing import Callable


class Solver:
    """Collections of differential equations solver methods as methods of
    'Solvers' instances.

    Attributes
    ----------
    func: Callable, default=None
        Function that returns the derivative of
        the chosen system.

    y0: np.ndarray, shape=(1, n), default=None
        Initial condition/state of the system.

    self.ts: np.ndarray, list of the times where to
        calculate the equation will be evaluated
    """

    def __init__(self, func: Callable, y0: np.ndarray, ts: np.ndarray,
                 args=()) -> None:
        """Associating attributes defined in class definition. This function
        is used to initialise the given system.
        """
        self.y0 = np.array(y0)
        self.func = func
        self.ts = np.array(ts)
        self.args = args

        ys = np.zeros(shape=(len(ts), y0.shape[0]))
        ys[0, :] = y0

    def euler(self) -> np.ndarray:
        """Euler method implementation.

        Returns
        -------
        _: np.ndarray, shape=(len(self.y0), len(self.ts))
            the state of the system at every time self.ts solved with
            the euler method
        """
        ys = np.zeros(shape=(len(self.ts), len(self.y0)))
        ys[0, :] = self.y0
        y = self.y0.copy()
        dts = self.ts[1:] - self.ts[:-1]
        for idx, dt in enumerate(dts):
            y = y + dt * self.func(y, self.ts[idx], *self.args)
            ys[idx + 1] = y

        return ys

    def pred_corr(self) -> np.ndarray:
        """predicteur correcteur method implementation.

        Returns
        -------
        _: np.ndarray, shape=(len(self.y0), len(self.ts))
            the state of the system at every time self.ts solved with
            the predicteur correcteur method
        """
        ys = np.zeros(shape=(len(self.ts), len(self.y0)))
        ys[0, :] = self.y0
        y = self.y0.copy()
        dts = self.ts[1:] - self.ts[:-1]
        for idx, dt in enumerate(dts):
            k1 = dt * self.func(y, self.ts[idx], *self.args)
            k2 = dt * self.func(y + k1, self.ts[idx + 1], *self.args)
            y = y + (k1 + k2) / 2
            ys[idx + 1] = y

        return ys

    def runge_kutta(self, order: int = 2) -> np.ndarray:
        """Runge-Kutta method implementation.

        Parameters
        ----------
        order: int, 2 or 4, default=2
            Order in 'dt' of the approximation.

        Returns
        -------
        _: np.ndarray, shape=(len(self.y0), len(self.ts))
            the state of the system at every time self.ts solved with
            the Runge-Kutta method
        """
        ys = np.zeros(shape=(len(self.ts), len(self.y0)))
        ys[0, :] = self.y0
        y = self.y0.copy()
        dts = self.ts[1:] - self.ts[:-1]
        if order == 2:
            for idx, dt in enumerate(dts):
                k1 = dt * self.func(y, self.ts[idx], *self.args)
                k2 = dt * self.func(y + k1 / 2,
                                    self.ts[idx] + dt / 2, *self.args)
                y = y + k2
                ys[idx + 1] = y

        elif order == 4:
            for idx, dt in enumerate(dts):
                k1 = dt * self.func(y, self.ts, *self.args)
                k2 = dt * self.func(y + k1 / 2, self.ts + dt / 2, *self.args)
                k3 = dt * self.func(y + k2 / 2, self.ts + dt / 2, *self.args)
                k4 = dt * self.func(y + k3, self.ts + dt, *self.args)
                y = y + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
                ys[idx + 1] = y

        return ys

    def scipy_solver(self) -> np.ndarray:
        """Scipy method from 'scipy.integrate.odeint' function.

        Parameters
        ----------
        order: int, 2 or 4, default=2
            Order in 'dt' of the approximation.

        Returns
        -------
        _: np.ndarray, shape=(len(self.y0), len(self.ts))
            the state of the system at every time self.ts solved with
            'scipy.integrate.odeint'
        """
        ys = sp.integrate.odeint(
            func=self.func,
            y0=self.y0,
            t=self.ts,
            args=self.args
        )
        return ys

    def compare_methods(self) -> dict:
        """Solves the given system using every methods.

        Returns
        -------
        _: dictionary of np.ndarray, shape=(len(self.y0), len(self.ts)):
            'euler': the euler resolution
            'pred corr': the prediction correction solution
            'runge kutta o2': the runge kutta of order 2 solution
            'runge kutta o4': the runge kutta of order 4 solution
            'scipy': the scipy.integrate.odeint solution
        """
        methods = {
            "euler": self.euler(),
            "pred corr": self.pred_corr(),
            "runge kutta o2": self.runge_kutta(order=2),
            "runge kutta o4": self.runge_kutta(order=4),
            "scipy": self.scipy_solver(),
        }
        return methods


if __name__ == "__main__":
    pass
