# 4_ising_bedard_delagrave

Ce dépôt contient tout les scripts nécessaires au 4e devoir de l'équipe
de Michael Bédard et Antoine de Lagrave sur la simulation Monte Carlo d'une
grille de spin bidimensionnelle.

## Rapport
Le rapport est le pdf nommé "Rapport_devoir_4.pdf" à la racine du projet

## Requirements

L'utilisation de ce projet via le programme `poetry` est fortement encouragée.
Il est toutefois obligatoire d'avoir une version du programme plus élevée que
1.1.13 pour la compatibilité des fichiers de configuration. Pour synchroniser
les dépendances nécessaires et fabriquer votre propre environnement virtuel à
l'aide de cet outil la commande

```shell
poetry install
```

executée à la racine du projet devrait suffire.

## Usage

Pour obtenir les mêmes graphiques présentés dans le rapport, il suffit
simplement d'exécuter le script `./ising/main.py` avec `poetry` en utilisant
la commande

```shell
poetry run python3 ./ising/main.py
```

### Description des fichiers

#### `./ising/figs`

Contient les graphiques/figures générés par le fichier python `main.py`.

#### `./ising/data`

Contient tous les fichiers de données '.txt' générés par la simulation,
c.-à.-d. les énergies, l'aimantation et les temps de corrélation associés
aux différentes températures de la simulation.

#### `./ising/ising.py`

Contient la définition des objets `Ising` (grille bidimensionnelle de spins)
ainsi que celle des objets `Observable` (mesures d'énergies et d'aimantation).

#### `./ising/simulate.py`

Contient l'implémentation de la simulation Monte Carlo ainsi que la fonction
permettant de mettre en graphique les résultats obtenus.

#### `./ising/main.py`

Permet d'appeler toutes les fonctions nécessaires à la simulation et la mise
en graphique des résultats via la fonction `main()`.
