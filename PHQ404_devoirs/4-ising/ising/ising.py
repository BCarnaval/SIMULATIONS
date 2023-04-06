#!/usr/bin/env python3
# -*- coding UFT-8 -*-
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def ising_aleatoire(temperature, taille):
    """ Génére une grille aléatoire de spins.

    Arguments
    ---------
    temperature : Température du système.
    taille : La grille a une dimension taille x taille.
    """
    # On initialise aléatoirement des spins de valeurs -1 ou +1
    # sur un grille de dimension taille x taille.
    spins = np.random.randint(0, 2, (taille, taille))
    spins = 2 * spins - 1
    return Ising(temperature, spins)


# Numba permet de compiler la classe pour qu'elle
# soit plus rapide. Il faut attention car certaines
# opérations ne sont plus permises.
@nb.experimental.jitclass([
    ("temperature", nb.float64),
    ("spins", nb.int64[:, :]),
    ("taille", nb.uint64),
    ("energie", nb.int64),
    ("aimantation", nb.int64)
])
class Ising:
    """Modèle de Ising paramagnétique en 2 dimensions.

    Représente une grille de spins classiques avec un couplage J = +1 entre
    les premiers voisins.

    Arguments
    ---------
    temperature : Température du système.
    spins : Tableau carré des valeurs de spins.
    """

    def __init__(self, temperature, spins):
        self.temperature = temperature
        self.spins = spins
        self.taille = np.shape(spins)[0]
        self.energie = self.calcule_energie()
        self.aimantation = self.calcule_aimantation()

    def difference_energie(self, x, y):
        """Retourne la différence d'énergie si le spin à la position (x, y)
        était renversé.
        """
        energy0 = 0
        n = self.taille

        # Local energy at position (x, y)
        energy0 += self.spins[x, y] * self.spins[(x + 1) % n, y]
        energy0 += self.spins[x, y] * self.spins[x, (y + 1) % n]
        energy0 += self.spins[x, y] * self.spins[(x - 1) % n, y]
        energy0 += self.spins[x, y] * self.spins[x, (y - 1) % n]

        # Flip spin
        self.spins[x, y] *= -1

        # Local energy at position (x, y)
        energy0 -= self.spins[x, y] * self.spins[(x + 1) % n, y]
        energy0 -= self.spins[x, y] * self.spins[x, (y + 1) % n]
        energy0 -= self.spins[x, y] * self.spins[(x - 1) % n, y]
        energy0 -= self.spins[x, y] * self.spins[x, (y - 1) % n]

        # Back to initial configuration
        self.spins[x, y] *= -1

        return energy0

    def iteration_aleatoire(self):
        """Renverse un spin aléatoire avec probabilité ~ e^(-ΔE / T).

        Cette fonction met à jour la grille avec la nouvelle valeur de spin.
        """
        # Random site (x, y)
        i, j = np.random.randint(low=0, high=self.taille, size=2)
        delta_energy = self.difference_energie(x=i, y=j)
        if delta_energy <= 0:
            # Flip spin
            self.spins[i, j] *= -1

            # Update 'energie' attribute accordingly
            self.energie += delta_energy

        elif delta_energy > 0:
            pointer = np.random.rand()
            prob = np.exp(-delta_energy / self.temperature)
            if pointer <= prob:
                # Flip spin
                self.spins[i, j] *= -1

                # Update 'energie' attribute accordingly
                self.energie += delta_energy

        return

    def simulation(self, nombre_iterations):
        """Simule le système en effectuant des itérations aléatoires.
        """
        for _ in range(nombre_iterations):
            self.iteration_aleatoire()

    def calcule_energie(self):
        """Retourne l'énergie actuelle de la grille de spins.
        """
        energie = 0
        n = self.taille
        for x in range(n):
            for y in range(n):
                energie -= self.spins[x, y] * self.spins[(x + 1) % n, y]
                energie -= self.spins[x, y] * self.spins[x, (y + 1) % n]

        return energie

    def calcule_aimantation(self):
        """Retourne l'aimantation actuelle de la grille de spins.
        """
        return np.abs(self.spins.sum())


class Observable:
    """Utilise la méthode du binning pour calculer des statistiques
    pour un observable.

    Arguments
    ---------
    nombre_niveaux : Le nombre de niveaux pour l'algorithme. Le nombre de
                     mesures est exponentiel selon le nombre de niveaux.
    """

    def __init__(self, nombre_niveaux):
        self.nombre_niveaux = nombre_niveaux

        # Les statistiques pour chaque niveau
        self.nombre_valeurs = np.zeros(nombre_niveaux + 1, int)
        self.sommes = np.zeros(nombre_niveaux + 1)
        self.sommes_carres = np.zeros(nombre_niveaux + 1)

        # La dernière valeur ajoutée à chaque niveau.
        self.valeurs_precedentes = np.zeros(nombre_niveaux + 1)

        # Le niveau que nous allons utiliser.
        # La différence de 6 donne de bons résultats.
        # Voir les notes de cours pour plus de détails.
        self.niveau_erreur = self.nombre_niveaux - 6

    def ajout_mesure(self, valeur, niveau=0):
        """Ajoute une mesure.

        Arguments
        ---------
        valeur : Valeur de la mesure.
        niveau : Niveau auquel ajouter la mesure. Par défaut,
                 le niveau doit toujours être 0. Les autres niveaux
                 sont seulement utilisé pour la récursion.
        """
        self.nombre_valeurs[niveau] += 1
        self.sommes[niveau] += valeur
        self.sommes_carres[niveau] += valeur**2

        # Si un nombre pair de valeurs a été ajouté, on peut faire
        # une simplification.
        if self.nombre_valeurs[niveau] % 2 == 0:
            moyenne = (valeur + self.valeurs_precedentes[niveau]) / 2
            self.ajout_mesure(moyenne, niveau + 1)
        else:
            self.valeurs_precedentes[niveau] = valeur

        return

    def est_rempli(self) -> bool:
        """Retourne vrai si le binnage est complété.
        """
        return True if self.sommes[self.nombre_niveaux] != 0.0 else False

    def erreur(self) -> float:
        """Retourne l'erreur sur la mesure moyenne de l'observable.

        Le dernier niveau doit être rempli avant d'utiliser cette fonction.
        """
        erreurs = np.zeros(self.nombre_niveaux + 1)
        erreurs[self.niveau_erreur] = np.sqrt(
            (
                self.sommes_carres[self.niveau_erreur]
                - self.sommes[self.niveau_erreur]**2 /
                self.nombre_valeurs[self.niveau_erreur]
            ) / (
                self.nombre_valeurs[self.niveau_erreur]
                * (self.nombre_valeurs[self.niveau_erreur] - 1)
            )
        )
        return erreurs[self.niveau_erreur]

    def variance(self, niveau: int) -> float:
        """Docs
        """
        N = self.nombre_valeurs[niveau]
        var = 1/(N-1) * \
            (self.sommes_carres[niveau] - self.sommes[niveau]**2 / N)

        return var

    def temps_correlation(self):
        """Retourne le temps de corrélation.
        """
        erreur_0 = self.variance(0)
        erreur_inf = self.variance(self.niveau_erreur)

        return 1/2 * ((erreur_inf / erreur_0) - 1)

    def moyenne(self):
        """Retourne la moyenne des mesures.
        """
        return self.sommes[-1]


if __name__ == '__main__':
    pass
