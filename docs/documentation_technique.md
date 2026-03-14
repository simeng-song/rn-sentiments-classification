# Documentation technique

## 1. Objectifs du projet

Dans ce projet, nous avons voulu comparer les sentiments dans deux sous-corpus Amazon : `Books` et `Kindle`(e-books).

On s'est posé deux questions principales :

- est-ce qu'on observe des différences entre les avis sur les livres papier et les livres numériques ?
- est-ce que la classification en trois classes avec `neutre` rend la tâche plus difficile ?

Pour répondre à ces questions, nous avons comparé deux modèles :

- une baseline avec `TF-IDF + régression logistique`
- un modèle neuronal avec `DistilBERT`

DistilBERT est le modèle principal de notre travail.

## 2. Données utilisées

Les données viennent du corpus **Amazon Reviews 2023** :
https://amazon-reviews-2023.github.io/

Comme ce corpus est très grand, nous avons travaillé sur des échantillons pour que le projet reste faisable sur nos ordinateurs.
Dans notre plan de travail, nous avions choisi de garder environ `10 000` avis par sous-corpus pour éviter des temps de calcul trop longs, surtout pour l'entraînement local.

On a utilisé plusieurs formats :

- `jsonl` pour les données de départ ;
- `jsonl` nettoyé avec les champs ajoutés ;
- `csv` pour l'entraînement et l'évaluation.

Au niveau juridique, on utilise ces données seulement dans le cadre du projet de cours. Le dépôt contient surtout des échantillons et des fichiers de travail.

## 3. Prétraitement

Le prétraitement suit plusieurs étapes :

1. prendre un échantillon du corpus ;
2. nettoyer le texte ;
3. créer un label à partir de la note ;
4. extraire les colonnes utiles ;
5. diviser les données en `train`, `dev` et `test`.

Pour la division des données, nous avons gardé une répartition classique :

- `train` : `80 %`
- `dev` : `10 %`
- `test` : `10 %`

Pour les labels, on a choisi la logique suivante :

- notes `1` et `2` : `negative`
- note `3` : `neutre`
- notes `4` et `5` : `positive`

On a aussi gardé la possibilité de faire une expérience sans la classe `neutre`, pour comparer les deux cas.
Ce choix était important dans notre plan de départ, parce qu'on voulait justement voir si la classe `neutre` apportait une information utile ou si elle rendait surtout la classification plus difficile.

## 4. Méthodologie

Au début, notre projet était surtout un ensemble de scripts séparés. Ensuite, on a réorganisé le dépôt pour avoir une structure plus claire, avec une interface en ligne de commande.

Les problèmes principaux qu'on a rencontrés étaient :

- des scripts avec des chemins en dur ;
- une organisation un peu dispersée ;
- une documentation trop faible ;
- peu de tests.

Pour améliorer cela, nous avons :

- regroupé la logique principale dans `src/amazon_sentiment/` ;
- ajouté une CLI unique ;
- organisé les sorties dans des dossiers plus clairs ;
- ajouté quelques tests simples ;
- réécrit la documentation en français plus simple.

## 5. Répartition du travail

Le projet a été réalisé en binôme.

- `Simeng SONG` : préparation des données, nettoyage, création des labels, division en jeux de données, première exploration.
- `Xiaobo WANG` : entraînement des modèles, évaluation, visualisation, mise en place de la structure du projet.

Nous avons aussi travaillé ensemble sur :

- le choix du sujet ;
- l'analyse des résultats ;
- la rédaction finale.

Concrètement, cette répartition correspond surtout aux fichiers suivants :

- pour la préparation des données : `src/amazon_sentiment/preprocessing.py`, `src/amazon_sentiment/labeling.py` et `src/amazon_sentiment/dataset.py`
- pour l'entraînement et l'évaluation : `src/amazon_sentiment/models.py` et `src/amazon_sentiment/evaluation.py`
- pour la visualisation et la prédiction : `src/amazon_sentiment/visualize.py` et `src/amazon_sentiment/predict.py`
- pour l'utilisation globale du projet en ligne de commande : `src/amazon_sentiment/cli.py`

## 6. Implémentation

Le projet utilise surtout :

- Python
- pandas
- scikit-learn
- transformers
- datasets
- torch
- matplotlib
- nltk

Les fichiers principaux sont :

- `src/amazon_sentiment/preprocessing.py`
- `src/amazon_sentiment/labeling.py`
- `src/amazon_sentiment/dataset.py`
- `src/amazon_sentiment/models.py`
- `src/amazon_sentiment/evaluation.py`
- `src/amazon_sentiment/visualize.py`
- `src/amazon_sentiment/predict.py`
- `src/amazon_sentiment/cli.py`

On peut lancer le projet avec :

```bash
python -m src.amazon_sentiment --help
```

Dans notre plan initial, on voulait déjà comparer un modèle simple et un modèle plus fort. C'est pour cela qu'on a gardé :

- `TF-IDF + régression logistique` comme baseline ;
- `DistilBERT` comme modèle neuronal principal.

On a aussi choisi d'évaluer les modèles sur `dev` puis sur `test`, afin d'avoir une comparaison plus propre entre les différentes configurations.

## 7. Résultats et discussion

À la fin du projet, nous avons obtenu plusieurs types de sorties :

- des fichiers de résultats pour chaque expérience ;
- des métriques comme l'accuracy et le F1-score macro ;
- des matrices de confusion ;
- des graphiques comparatifs ;
- des fichiers de prédiction.

Les sorties se trouvent surtout dans deux ensembles complémentaires :

- `resultats/`, qui regroupe des fichiers de résultats et des visualisations utiles pour l'analyse ;
- `outputs/metrics/`, pour les métriques exportées ; (on n'a que pousse ce fichier sur github)
- `outputs/models/`, pour les modèles sauvegardés localement lors de l'entraînement ;
- `outputs/plots/`, pour les graphiques générés ;
- `outputs/predictions/`, pour les fichiers de prédiction.

PS : Pour la version déposée sur GitHub, nous n'avons pas laissé les gros fichiers de modèles entraînés dans le dépôt distant, parce que certains fichiers produits pendant l'entraînement dépassaient la limite de taille autorisée par GitHub. En revanche, nous avons conservé :

- le code complet pour réentraîner les modèles ;
- les fichiers de métriques ;
- les sorties d'analyse ;
- la structure des dossiers attendus.

Autrement dit, le dépôt permet de reproduire les expériences, même si les poids des modèles ne sont pas tous versionnés directement dans GitHub. Sur GitHub, on retrouve surtout les métriques exportées et les fichiers de résultats, tandis que les gros modèles restent disponibles localement après entraînement.

Notre comparaison porte sur quatre dimensions :

- `Books` contre `Kindle`
- régression logistique contre DistilBERT
- classification binaire contre classification à trois classes
- présence ou non de la classe `neutre`

Les résultats montrent d'abord que la régression logistique donne une baseline utile et assez rapide à entraîner. Elle permet d'avoir une première comparaison claire entre les deux corpus.

Mais dans l'ensemble, DistilBERT reste le modèle le plus intéressant dans notre projet. Même si son entraînement est plus coûteux, il prend mieux en compte le contexte des phrases, ce qui est important pour des avis clients où le sentiment n'est pas toujours exprimé de manière directe.

On observe aussi que la classification à trois classes est plus difficile que la classification binaire. La classe `neutre` rend les frontières entre les catégories moins nettes, et cela a souvent un effet négatif sur le F1-score macro. Ce point confirme notre question de départ : la classe `neutre` apporte une information utile, mais elle complique la tâche de classification.

La comparaison entre `Books` et `Kindle` nous a aussi permis de voir que, même si les deux sous-corpus sont proches, ils ne se comportent pas exactement de la même manière. C'était justement un des intérêts du projet : travailler sur deux catégories comparables, mais pas identiques.

Enfin, les visualisations nous ont aidés à rendre les comparaisons plus lisibles. Les graphiques d'accuracy et de F1-score macro permettent de voir plus rapidement les écarts entre modèles et entre configurations.
Cette partie faisait aussi déjà partie du plan du projet : on ne voulait pas seulement afficher des scores dans des fichiers texte, mais aussi produire des graphiques pour comparer plus facilement les expériences.

Si le projet avait pu être prolongé, nous aurions souhaité approfondir plusieurs points : une analyse d'erreurs plus détaillée avec examen des exemples mal classés, un réglage fin des hyperparamètres de DistilBERT, et un développement plus poussé de la partie extraction d'information.

## 8. Tests

Nous avons ajouté des tests simples pour vérifier plusieurs parties importantes du projet :

- la création des labels à partir des notes ;
- le nettoyage du texte ;
- la création des fichiers `train/dev/test`.

Ces tests se trouvent dans le dossier `tests/` :

- `tests/test_labeling.py`
- `tests/test_preprocessing.py`
- `tests/test_dataset.py`

L'objectif n'était pas de construire une suite de tests très complète, mais plutôt de vérifier que les étapes de base du pipeline fonctionnent correctement.

Dans le cadre d'une continuation du projet, plusieurs extensions seraient pertinentes : des tests pour la CLI, des tests vérifiant la génération des fichiers de métriques, et des tests de bout en bout pour la prédiction sur une phrase isolée ou un fichier CSV complet.
