# Projet de prédiction VO2max

Ce projet vise à développer des modèles d'apprentissage automatique pour prédire la VO2max (consommation maximale d'oxygène) à partir de données d'activité physique.

## Structure du projet

```
├── data/
│   ├── mipace_mlproject.csv      # Jeu de données principal
│   └── antoine/                  # Données personnelles pour prédiction
│       ├── activity.csv          # Données brutes d'activité
│       ├── convert_to_mipace_format.py   # Script de conversion des données
│       ├── predict_vo2max.py     # Script de prédiction du VO2max
│       └── results/              # Résultats des prédictions
├── models/                       # Scripts des modèles d'apprentissage
│   ├── vo2max_prediction_mlp.py  # Modèle de réseau de neurones
│   └── vo2max_prediction_svm.py  # Modèle SVM
└── resultats/                    # Résultats des entraînements et évaluations
    ├── MLP/                      # Résultats du modèle MLP
    └── SVM/                      # Résultats du modèle SVM
```

## Fonctionnalités

- Entraînement de modèles pour prédire la VO2max :
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (MLP)
- Analyse de l'importance des caractéristiques
- Conversion de données d'activité en format compatible
- Prédiction de VO2max sur de nouvelles données

## Caractéristiques utilisées

Le modèle utilise les caractéristiques suivantes :
- Données physiologiques : âge, poids, taille, IMC, fréquence cardiaque maximale, sexe
- Données d'activité : vitesse maximale, fréquence cardiaque moyenne des 30 dernières minutes
- Données environnementales : température ambiante, humidité, présence de pente
- Caractéristiques avancées (optionnelles) : économie de course (avgRE), volume d'éjection systolique estimé (eSV)

## Utilisation

### Entraînement des modèles

```bash
# Entraînement du modèle SVM
python models/vo2max_prediction_svm.py

# Entraînement du modèle MLP
python models/vo2max_prediction_mlp.py
```

### Préparation et prédiction sur de nouvelles données

1. Convertir les données d'activité au format requis :
```bash
python data/antoine/convert_to_mipace_format.py
```

2. Effectuer des prédictions avec les modèles entraînés :
```bash
python data/antoine/predict_vo2max.py
```

## Résultats

Les résultats des modèles incluent :
- Graphiques de comparaison entre valeurs réelles et prédites
- Analyse de l'importance des caractéristiques
- Métriques de performance (RMSE, R²)
- Prédictions individuelles pour les nouvelles données

## Performances des modèles

Les deux modèles (SVM et MLP) affichent de bonnes performances pour la prédiction de la VO2max, avec des scores R² généralement supérieurs à 0.80, indiquant une bonne capacité prédictive.
