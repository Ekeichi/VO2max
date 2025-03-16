#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'resultats', 'MLP')

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

# Encodage des variables catégorielles
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

# Création et entraînement du modèle MLP (Multilayer Perceptron)
# Recherche d'hyperparamètres optimaux
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],  # Adam est généralement plus efficace
    'alpha': [0.0001, 0.001, 0.01, 0.1],  # Ajout d'une valeur plus élevée pour la régularisation
    'learning_rate_init': [0.001, 0.01],  # Taux d'apprentissage initial
    'learning_rate': ['adaptive'],  # Taux d'apprentissage adaptatif
    'max_iter': [2000],  # Augmentation du nombre d'itérations
    'early_stopping': [True],  # Arrêt anticipé
    'n_iter_no_change': [10],  # Nombre d'itérations sans amélioration pour l'arrêt anticipé
    'tol': [1e-4]  # Tolérance pour la convergence
}

# Ajouter un message pour suivre l'avancement
print("Démarrage de la recherche d'hyperparamètres pour le MLP...")

# Ajouter une stratégie de validation avec un ensemble de validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

# Supprimer les avertissements de convergence pendant la recherche d'hyperparamètres
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

mlp_regressor = MLPRegressor(random_state=42, verbose=0)
grid_search = GridSearchCV(
    mlp_regressor, 
    param_grid, 
    cv=3,  # Réduction du nombre de plis pour accélérer
    scoring='neg_mean_squared_error', 
    n_jobs=-1,
    verbose=1  # Afficher la progression
)

# Entraînement avec les données
print("Entraînement du modèle MLP en cours...")
grid_search.fit(X_train_scaled, y_train)

# Réactiver les avertissements après la recherche
warnings.resetwarnings()

print(f"Meilleur score de validation: {-grid_search.best_score_:.4f} MSE")

# Meilleur modèle
best_mlp = grid_search.best_estimator_

# Prédictions sur l'ensemble de test
y_pred = best_mlp.predict(X_test_scaled)

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
plt.title('Comparaison entre valeurs réelles et prédites (MLP)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'vo2max_prediction_mlp_results.png'))

# Courbe d'apprentissage (historique de perte)
if hasattr(best_mlp, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(best_mlp.loss_curve_)
    plt.title('Courbe d\'apprentissage')
    plt.xlabel('Itérations')
    plt.ylabel('Perte (MSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'mlp_learning_curve.png'))

# Analyse de l'importance des caractéristiques via un modèle permutation
from sklearn.inspection import permutation_importance

print("Calcul de l'importance des caractéristiques...")
result = permutation_importance(
    best_mlp, 
    X_test_scaled, 
    y_test, 
    n_repeats=10, 
    random_state=42,
    n_jobs=-1  # Utiliser tous les processeurs disponibles
)
importance = result.importances_mean
feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Afficher les caractéristiques les plus importantes
print("\nImportance des caractéristiques:")
for i, (feature, importance) in enumerate(zip(feature_importance['Feature'].values, 
                                             feature_importance['Importance'].values)):
    if i < 5:  # Afficher les 5 plus importantes
        print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Importance des caractéristiques (permutation)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'mlp_feature_importance.png'))

print(f"\nAnalyse terminée. Les résultats ont été enregistrés dans {RESULTS_DIR}.")
print(f"Meilleur modèle: {grid_search.best_params_}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")
