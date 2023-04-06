#!/usr/bin/env python3
# -*- coding UFT-8 -*-
"""This module aims to be the one each user should call to use the program.
"""
import numpy as np
import matplotlib.pyplot as plt

from bouali import Bouali
from lorenz import Lorenz
from rossler import Rossler
from attractor import ChaosAttractor


def main() -> None:
    """Docs
    """
    r0 = np.array([0, 1, -1])
    times = np.linspace(0, 100, 100000)

    master_b = ChaosAttractor(
        name='general bouali',
        init_state=np.array([0.2, 0.2, -0.1]),
        time_domain=times
    )
    B = Bouali(attractor=master_b, alpha=3.0, beta=2.2, gamma=1, mu=1.5e-3)

    master_l = ChaosAttractor(
        name='general lorenz',
        init_state=r0,
        time_domain=times
    )
    L = Lorenz(attractor=master_l, sigma=10, rho=28, beta=8/3)

    master_r = ChaosAttractor(
        name='general Rossler',
        init_state=r0,
        time_domain=times
    )
    Rossler(attractor=master_r, alpha=0.2, beta=0.2, gamma=5.7)

    # Plot solutions
    s, n = 25, len(B)
    sol = L.compute_trajectory()

    cmap = plt.cm.GnBu
    x, y, z = sol[:, 0], sol[:, 1], sol[:, 2]

    ax = plt.figure().add_subplot(projection='3d')

    for i in range(0, n - s, s):
        ax.plot(x[i: i + s + 1], y[i: i + s + 1],
                z[i: i + s + 1], color=cmap(i/n))

    plt.show()

    return


if __name__ == "__main__":
    main()
