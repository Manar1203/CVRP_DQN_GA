# CVRP_GA : Résolution du Problème de Tournées de Véhicules (CVRP) avec un Algorithme Génétique Pur

Ce dépôt contient l'implémentation d'un algorithme génétique (AG) classique pour la résolution du problème de tournées de véhicules avec contraintes de capacité (CVRP). Ce projet sert de base et de point de comparaison pour des approches d'optimisation plus avancées.

## Fonctionnalités

* Lecture et/ou téléchargement automatique des fichiers d'instances VRP (au format TSPLIB).
* Implémentation des opérateurs génétiques standards (sélection par tournoi, croisement PMX, mutation par swap).
* Calcul de la distance totale des tournées.
* Visualisation des tournées optimisées et de la courbe de convergence de l'algorithme.

## Comment exécuter le code

1.  **Cloner le dépôt (si vous ne l'avez pas déjà fait) :**
    ```bash
    git clone [https://github.com/VOTRE_NOM_UTILISATEUR/CVRP_GA.git](https://github.com/VOTRE_NOM_UTILISATEUR/CVRP_GA.git)
    cd CVRP_GA
    ```
2.  **Installer les dépendances Python :**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Exécuter l'algorithme :**
    ```bash
    python pure_ga.py
    ```
    (Assurez-vous d'ajuster le nom du fichier Python si vous l'avez renommé différemment, ex: `python ga_vrp_solver.py`)

## Dépendances

Les dépendances requises sont listées dans `requirements.txt`. Vous pouvez les installer avec `pip`:






numpy
matplotlib



*(Note : `re`, `os`, `urllib.request`, `csv` sont des modules intégrés à Python et n'ont pas besoin d'être listés.)*

## Auteur
FAHIM Manar

