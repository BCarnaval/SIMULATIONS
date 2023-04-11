#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""This module aims to be the one each user should call to use the program.
"""
from bouali import Bouali
from lorenz import Lorenz
from rossler import Rossler


def main() -> None:
    """Docs
    """
    # Test using Lorenz system
    L = Lorenz()
    L.plot_trajectory()
    L.compute_lyapunovs()

    # Test using Rossler system
    R = Rossler()
    R.plot_trajectory()
    R.compute_lyapunovs()

    # Test using Bouali system
    B = Bouali()
    B.plot_trajectory()
    B.compute_lyapunovs()

    return


if __name__ == "__main__":
    main()
