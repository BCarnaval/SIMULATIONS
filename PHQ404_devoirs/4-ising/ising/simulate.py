# !/usr/bin/env python3
# -*- coding UFT-8 -*-
"""Simulation module
"""
import numpy as np
import matplotlib.pyplot as plt

from ising import Observable, ising_aleatoire


DATAPATH = './ising/data/'
FIGPATH = './ising/figs/'
# DATAPATH = 'data/'
# FIGPATH = 'figs/'


def simulate(temperatures: np.ndarray, update: int = 1000, levels: int = 16,
             warmup: int = 1000000, grid_size: int = 32,
             save: bool = False) -> None:
    """Simulates Ising model spins grid using Monte Carlo algorithm.

    Parameters
    ----------
    temperatures: np.ndarray, shape=(n,), default=None
        Temperatures at which simulate the system.

    update: int, default=1000
        Interval treshold to collect data in 'binning' method.

    levels: int, default=16
        Number of levels in 'binning' method.

    warmup: int, default=1000000
        Number of random iterations at which we consider that the spin
        distribution follows Bolztmann distribution.

    grid_size: int, default=32
        Size of on side of 2D spins grid (so grid_size**2 is the total number
        of spins).

    save: bool, default=False
        If set to True, saves energies, magnetization and correlation time data
        as text files in directory DATAPATH.
    """
    # Initializing data arrays
    energy_array = np.zeros(shape=(len(temperatures), 3))
    magn_array = np.zeros(shape=(len(temperatures), 3))
    tau_array = np.zeros(shape=(len(temperatures), 3))

    ising = ising_aleatoire(temperature=4.0, taille=grid_size)
    for idx, t in enumerate(temperatures):
        # Init ising object using random spins grid
        ising.temperature = t

        # Init energy & magnetization observables
        E = Observable(nombre_niveaux=levels)
        M = Observable(nombre_niveaux=levels)

        # Warmup system
        ising.simulation(warmup)
        print("Warmup...")

        while True:
            # Collect observables each 'update'
            ising.simulation(update)

            E.ajout_mesure(ising.calcule_energie())
            M.ajout_mesure(ising.calcule_aimantation())

            if E.est_rempli() and M.est_rempli():
                # Acess mean values of observables (for each spin)
                # so we deivide by the number of spins : spin_size**2
                energy_array[idx] = [
                    t,
                    E.moyenne() / grid_size**2,
                    E.erreur() / grid_size**2
                ]
                magn_array[idx] = [
                    t,
                    M.moyenne() / grid_size**2,
                    M.erreur() / grid_size**2
                ]
                tau_array[idx] = [
                    t,
                    E.temps_correlation(),
                    M.temps_correlation()
                ]

                print(f'Iteration : T = {t:.2f} ok!\n')
                break

    if save:
        save_path = DATAPATH

        # Energy data
        np.savetxt(fname=f'{save_path}energy.txt',
                   X=energy_array,
                   header='t E dE')

        # Magnetization data
        np.savetxt(fname=f'{save_path}magnetization.txt',
                   X=magn_array,
                   header='t M dM')

        # Correlation time data
        np.savetxt(fname=f'{save_path}temps_correlation.txt',
                   X=tau_array,
                   header='t tau_E tau_M')

    return


def plot_data(data_path: str = DATAPATH,
              save_figs: bool = False) -> None:
    """Plots energies, magnetization and correlation time using saved textfiles
    stored in './ising/data' directory.

    Parameters
    ----------
    filespath: str, default='./ising/data/'
        Directory in which we store data textfiles by default.

    save_figs: bool, default=False
        If set to True, saves plots in predetermined directory './ising/figs/'.
    """
    # Global attributes & variables
    save_path = FIGPATH

    # Retreive arrays
    energy_array = np.loadtxt(f'{data_path}energy.txt')
    magn_array = np.loadtxt(f'{data_path}magnetization.txt')
    tau_array = np.loadtxt(f'{data_path}temps_correlation.txt')

    # Defining obervables
    T = energy_array[:, 0]
    E, dE = energy_array[:, 1], energy_array[:, 2]
    M, dM = magn_array[:, 1], magn_array[:, 2]
    tau_E, tau_M = tau_array[:, 1], tau_array[:, 2]

    # Plot errorbars (E +/- dE & M +/- dM)
    plt.errorbar(T, E, fmt='-^', yerr=dE, label='Energy')
    plt.errorbar(T, M, fmt='-s', yerr=dM, label='Magnetization')
    plt.xlabel('Temperatures')
    plt.ylabel('Means')
    plt.legend()
    plt.tight_layout()

    if save_figs:
        plt.savefig(save_path + 'means.png', dpi=1200)

    plt.show()

    # Plot (tau E & tau M)
    plt.plot(T, tau_E, '-^', label=r'Energy')
    plt.plot(T, tau_M, '-s', label=r'Magnetization')
    plt.xlabel('Temperatures')
    plt.ylabel('Correlation time')
    plt.legend()
    plt.tight_layout()

    if save_figs:
        plt.savefig(save_path + 'taus.png', dpi=1200)

    plt.show()

    return


if __name__ == "__main__":
    plot_data(save_figs=True)
