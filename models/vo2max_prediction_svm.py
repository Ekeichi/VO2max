#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultats', 'SVM')

# Créer le dossier de résultats s'il n'existe pas
os.makedirs(RESULTS_DIR, exist_ok=True)

# Charger les données
df = pd.read_csv(os.path.join(DATA_DIR, 'mipace_mlproject.csv'))

# Prétraitement des données
# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Sélectionner les caractéristiques pertinentes pour prédire VO2max
features = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
            'RH Humidity', 'avgHR last 30', 'MaxSpeed', 'avgRE', 'eSV']

# Encodage de la variable catégorielle
df['gender_encoded'] = df['gender'].map({'Male': 0, 'Female': 1})
df['Grade_encoded'] = df['Grade'].map({True: 1, False: 0})

# Ajouter les variables encodées aux caractéristiques
features.extend(['gender_encoded', 'Grade_encoded'])

# Séparer les caractéristiques (X) et la cible (y)
X = df[features]
y = df['V02max']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Création et entraînement du modèle SVM
# Recherche d'hyperparamètres optimaux
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly']
}



svm_regressor = SVR()
grid_search = GridSearchCV(svm_regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Meilleur modèle
best_svm = grid_search.best_estimator_

# Prédictions sur l'ensemble de test
y_pred = best_svm.predict(X_test_scaled)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Meilleurs hyperparamètres: {grid_search.best_params_}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('VO2max réelle')
plt.ylabel('VO2max prédite')
plt.title('Comparaison entre valeurs réelles et prédites')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'vo2max_prediction_results.png'))

# Analyse de l'importance des caractéristiques (pour le noyau linéaire)
if best_svm.kernel == 'linear':
    importance = np.abs(best_svm.coef_[0])
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title("Importance des caractéristiques")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'))

print("Analyse terminée. Les résultats ont été enregistrés.")
