# Analyse de sentiments sur des avis Amazon avec DistilBERT

## Présentation du projet

Dans ce projet, nous travaillons sur l'analyse de sentiments à partir d'avis Amazon. On a choisi deux catégories proches : `Books` et `Kindle`. L'idée est de voir si les sentiments exprimés dans les commentaires changent selon le type de produit, et aussi de comparer plusieurs façons de faire la classification.

Nous comparons deux approches :

- une méthode plus classique : `TF-IDF + régression logistique`
- une méthode avec réseau de neurones : `DistilBERT`

## Répartition du travail

Le projet a été réalisé en binôme.

- `Simeng SONG` : préparation des données, nettoyage, création des labels, division en `train/dev/test`, première exploration des données.
- `Xiaobo WANG` : entraînement des modèles, comparaison des résultats, évaluation, visualisation.

On a aussi travaillé ensemble sur :

- le choix du sujet ;
- l'interprétation des résultats ;
- la rédaction de la documentation finale.

## Objectifs

Nos objectifs sont les suivants :

- comparer les avis `Books` et `Kindle` ;
- comparer une classification en 2 classes et une classification en 3 classes ;
- voir si la classe `neutre` aide ou complique la tâche ;
- comparer une baseline simple avec un modèle neuronal plus fort.

## Données utilisées

Les données viennent du corpus **Amazon Reviews 2023** :

- https://amazon-reviews-2023.github.io/

Comme le corpus est très grand, on a travaillé sur des échantillons. Cela nous permet de faire les expériences sur nos machines sans temps de calcul trop long.

Formats utilisés :

- fichiers bruts en `jsonl`
- fichiers nettoyés et annotés en `jsonl`
- fichiers d'entraînement en `csv`

## Statut des données

On utilise ces données dans un cadre universitaire. Le dépôt contient seulement des échantillons et des fichiers de travail. Pour une réutilisation plus large, il faut vérifier les conditions du corpus source.

## Fonctionnement du projet

Le projet suit les étapes suivantes :

1. prendre un échantillon du corpus ;
2. nettoyer les textes ;
3. créer les labels à partir des notes ;
4. extraire les colonnes utiles ;
5. diviser les données en `train`, `dev` et `test` ;
6. entraîner les modèles ;
7. comparer les résultats ;
8. faire une prédiction sur une phrase ou sur un petit fichier CSV.

---

## Installation

```bash
pip install -r requirements.txt
```

## Commandes principales

Afficher l'aide :

```bash
python -m src.amazon_sentiment --help
```

### Prétraitement

```bash
python -m src.amazon_sentiment preprocess \
  --input-jsonl data/raw/Books_échantillon.jsonl \
  --clean-output data/interim/books_cleaned.jsonl \
  --labeled-output data/interim/books_labeled.jsonl \
  --csv-output data/processed/books_3class.csv \
  --split-output-dir data/processed/books_3class \
  --prefix books_3class \
  --include-neutral
```

### Entraînement de la régression logistique

```bash
python -m src.amazon_sentiment train-logreg \
  --train data/labeled/Books_3_dataset/Books_3_train.csv \
  --dev data/labeled/Books_3_dataset/Books_3_dev.csv \
  --test data/labeled/Books_3_dataset/Books_3_test.csv \
  --include-neutral
```

### Entraînement de DistilBERT

```bash
python -m src.amazon_sentiment train-bert \
  --train data/labeled/Books_3_dataset/Books_3_train.csv \
  --dev data/labeled/Books_3_dataset/Books_3_dev.csv \
  --test data/labeled/Books_3_dataset/Books_3_test.csv \
  --include-neutral \
  --epochs 2
```

### Évaluation

```bash
python -m src.amazon_sentiment evaluate --metrics-file outputs/metrics/distilbert_books_3classes.json
```

### Visualisation

```bash
python -m src.amazon_sentiment visualize \
  --metrics-files outputs/metrics/*.json \
  --summary-csv outputs/metrics/summary.csv \
  --output-dir outputs/plots
```

### Prédiction

Pour une phrase :

```bash
python -m src.amazon_sentiment predict \
  --model-type bert \
  --model-path outputs/models/distilbert_books_3classes \
  --text "Ce roman était très intéressant et bien écrit."
```

Pour un fichier CSV :

```bash
python -m src.amazon_sentiment predict \
  --model-type bert \
  --model-path outputs/models/distilbert_books_3classes \
  --input-csv data/sample/sample_reviews.csv \
  --text-column text \
  --output-csv outputs/predictions/sample_predictions.csv
```

## Sorties du projet

On trouve les sorties du projet dans deux ensembles complémentaires :

- `resultats/` contient des fichiers de résultats et des visualisations utilisés pour l'analyse comparative ;
- `outputs/models/` contient les modèles sauvegardés ;
- `outputs/metrics/` contient les métriques exportées ;
- `outputs/plots/` contient les graphiques générés ;
- `outputs/predictions/` contient les fichiers de prédiction.

Cette organisation nous permet de garder à la fois les résultats d'analyse et les sorties produites directement par l'application.

## Structure du dépôt

```text
rn_sentiments_classification/
├── README.md
├── requirements.txt
├── docs/
│   └── documentation_technique.md
├── data/
│   ├── raw/
│   ├── labeled/
│   └── sample/
├── outputs/
│   ├── metrics/
│   ├── models/
│   ├── plots/
│   └── predictions/
├── resultats/
├── src/
│   ├── __init__.py
│   └── amazon_sentiment/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── preprocessing.py
│       ├── labeling.py
│       ├── dataset.py
│       ├── models.py
│       ├── evaluation.py
│       ├── visualize.py
│       └── predict.py
└── tests/
```

## Documentation technique

On a réuni la documentation technique dans :

- `docs/documentation_technique.md`
