#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""This module defines general functions used in the caracterisation of chaos
attractor.
"""

import time
import numpy as np
from rich import print
from functools import wraps


def time_iters(func):
    """Wraper that prints the processing time of a given function"""

    @wraps(func)
    def time_iters_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} took {total_time:.4f}.")
        return result
    return time_iters_wrapper


def epsilon(serie: np.ndarray):
    """Computes the convergence of a series using the epsilon algorithm.

    Parameters
    ----------
    serie: list, tuple or np.array
        The series which we wish to converge.

    Returns
    -------
    _: float
        The convergence of the array.
    """
    serie = np.array(serie)
    if not serie.shape[0] % 2:
        serie = np.insert(serie, 0, np.array([0]))
    else:
        pass

    def _epsilon(s_0, s_1):
        """This function computes one step down the pyramid of
        the epsilon algorythm.
        """
        return s_1, s_0[1:-1] + 1/(s_1[1:] - s_1[:-1])

    s_0, s_1 = np.zeros(len(serie)+1), serie
    while len(s_1) != 1:
        s_0, s_1 = _epsilon(s_0, s_1)

    return s_1[0]


if __name__ == "__main__":
    pass
