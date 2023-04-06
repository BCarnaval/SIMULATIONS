# !/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Main module which any user should use to perform Monte Carlo simulation on
Ising model.
"""
import numpy as np

from simulate import simulate, plot_data


def main() -> None:
    """Runs simulation
    """
    # Generate data
    temps = np.arange(1, 4, 0.1)
    simulate(temperatures=temps, levels=16, save=True)

    # Display results
    plot_data(save_figs=True)

    return


if __name__ == "__main__":
    main()
