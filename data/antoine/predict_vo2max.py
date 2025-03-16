#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Définir les chemins
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

# Créer le dossier de résultats s'il n'existe pas
os.makedirs(RESULTS_DIR, exist_ok=True)

# Fonction pour entraîner et sauvegarder les modèles
def train_and_save_models(use_avgRE_eSV=False):
    print("Chargement des données d'entraînement...")
    # Charger les données d'entraînement
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'mipace_mlproject.csv'))
    
    # Prétraitement
    df_train = df_train.dropna()
    
    # Sélectionner les caractéristiques de base (sans avgRE et eSV)
    base_features = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                     'RH Humidity', 'avgHR last 30', 'MaxSpeed']
    
    # Encoder les variables catégorielles
    df_train['gender_encoded'] = df_train['gender'].map({'Male': 0, 'Female': 1})
    df_train['Grade_encoded'] = df_train['Grade'].map({True: 1, False: 0})
    
    # Ajouter les variables encodées
    base_features.extend(['gender_encoded', 'Grade_encoded'])
    
    # Déterminer quelles caractéristiques utiliser
    if use_avgRE_eSV:
        print("Entraînement avec avgRE et eSV inclus")
        features = base_features + ['avgRE', 'eSV']
    else:
        print("Entraînement sans avgRE et eSV")
        features = base_features
    
    # Séparer X et y
    X = df_train[features]
    y = df_train['V02max']
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardiser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Sauvegarder le scaler
    scaler_file = 'scaler_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'scaler_without_avgRE_eSV.pkl'
    joblib.dump(scaler, os.path.join(RESULTS_DIR, scaler_file))
    
    # Entraîner le modèle SVM
    print("Entraînement du modèle SVM...")
    svm_model = SVR(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X_train_scaled, y_train)
    
    # Sauvegarder le modèle SVM
    svm_file = 'svm_model_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'svm_model_without_avgRE_eSV.pkl'
    joblib.dump(svm_model, os.path.join(RESULTS_DIR, svm_file))
    
    # Entraîner le modèle MLP
    print("Entraînement du modèle MLP...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(100, 50), 
        activation='relu', 
        solver='adam',
        alpha=0.001,
        max_iter=2000, 
        early_stopping=True,
        random_state=42
    )
    mlp_model.fit(X_train_scaled, y_train)
    
    # Sauvegarder le modèle MLP
    mlp_file = 'mlp_model_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'mlp_model_without_avgRE_eSV.pkl'
    joblib.dump(mlp_model, os.path.join(RESULTS_DIR, mlp_file))
    
    print(f"Modèles entraînés et sauvegardés avec succès! (avec avgRE et eSV: {use_avgRE_eSV})")
    
    # Évaluer sur l'ensemble de test
    X_test_scaled = scaler.transform(X_test)
    svm_score = svm_model.score(X_test_scaled, y_test)
    mlp_score = mlp_model.score(X_test_scaled, y_test)
    
    print(f"Score SVM (R²): {svm_score:.4f}")
    print(f"Score MLP (R²): {mlp_score:.4f}")
    
    return features, scaler, svm_model, mlp_model

# Fonction pour prédire VO2max
def predict_vo2max(data_path, features, scaler, svm_model, mlp_model):
    print(f"Prédiction de VO2max pour les données dans {data_path}...")
    
    # Charger les données
    df_antoine = pd.read_csv(data_path)
    
    # Encoder les variables catégorielles si nécessaire
    df_antoine['gender_encoded'] = df_antoine['gender'].map({'Male': 0, 'Female': 1})
    df_antoine['Grade_encoded'] = df_antoine['Grade'].map({True: 1, False: 0})
    
    # Créer un DataFrame avec les caractéristiques nécessaires
    X_pred = df_antoine[features].copy()
    
    print(f"Caractéristiques utilisées pour la prédiction: {features}")
    
    # Standardiser les données
    X_pred_scaled = scaler.transform(X_pred)
    
    # Prédictions avec les deux modèles
    print("Prédiction avec le modèle SVM...")
    svm_predictions = svm_model.predict(X_pred_scaled)
    
    print("Prédiction avec le modèle MLP...")
    mlp_predictions = mlp_model.predict(X_pred_scaled)
    
    # Moyenne des prédictions (ensemble)
    ensemble_predictions = (svm_predictions + mlp_predictions) / 2
    
    # Ajouter les prédictions au DataFrame
    df_antoine['VO2max_SVM'] = svm_predictions
    df_antoine['VO2max_MLP'] = mlp_predictions
    df_antoine['VO2max_Ensemble'] = ensemble_predictions
    
    # Sauvegarder les résultats
    output_path = os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions.csv')
    df_antoine.to_csv(output_path, index=False)
    
    # Afficher les résultats
    print("\nRésultats des prédictions de VO2max:")
    for i, row in df_antoine.iterrows():
        print(f"ID: {row['id']}")
        print(f"  - VO2max (SVM): {row['VO2max_SVM']:.2f} ml/kg/min")
        print(f"  - VO2max (MLP): {row['VO2max_MLP']:.2f} ml/kg/min")
        print(f"  - VO2max (Ensemble): {row['VO2max_Ensemble']:.2f} ml/kg/min")
    
    # Créer un graphique comparatif
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(df_antoine))
    width = 0.25
    
    plt.bar(x - width, svm_predictions, width, label='SVM')
    plt.bar(x, mlp_predictions, width, label='MLP')
    plt.bar(x + width, ensemble_predictions, width, label='Ensemble')
    
    plt.xlabel('Activité')
    plt.ylabel('VO2max prédit (ml/kg/min)')
    plt.title('Prédictions de VO2max par différents modèles')
    plt.xticks(x, df_antoine['id'])
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(os.path.join(RESULTS_DIR, 'vo2max_predictions_comparison.png'))
    
    print(f"\nRésultats sauvegardés dans {output_path}")
    print(f"Graphique sauvegardé dans {os.path.join(RESULTS_DIR, 'vo2max_predictions_comparison.png')}")
    
    return df_antoine

# Programme principal
if __name__ == "__main__":
    # Créer deux types de modèles : un avec avgRE et eSV, et un sans
    
    # Vérifier si les modèles sans avgRE et eSV existent déjà
    svm_model_path = os.path.join(RESULTS_DIR, 'svm_model_without_avgRE_eSV.pkl')
    mlp_model_path = os.path.join(RESULTS_DIR, 'mlp_model_without_avgRE_eSV.pkl')
    scaler_path = os.path.join(RESULTS_DIR, 'scaler_without_avgRE_eSV.pkl')
    
    if os.path.exists(svm_model_path) and os.path.exists(mlp_model_path) and os.path.exists(scaler_path):
        print("Chargement des modèles existants (sans avgRE et eSV)...")
        scaler = joblib.load(scaler_path)
        svm_model = joblib.load(svm_model_path)
        mlp_model = joblib.load(mlp_model_path)
        
        # Liste des caractéristiques utilisées pour la prédiction (sans avgRE et eSV)
        features = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                   'RH Humidity', 'avgHR last 30', 'MaxSpeed',
                   'gender_encoded', 'Grade_encoded']
    else:
        print("Entraînement de nouveaux modèles (sans avgRE et eSV)...")
        # Entraîner le modèle sans avgRE et eSV
        features, scaler, svm_model, mlp_model = train_and_save_models(use_avgRE_eSV=False)
        
        # Pour comparer, entraîner aussi un modèle avec avgRE et eSV
        print("\nPour comparaison, entraînement de modèles avec avgRE et eSV...")
        features_with_avgRE_eSV, _, _, _ = train_and_save_models(use_avgRE_eSV=True)
    
    # Chemin vers les données d'Antoine
    antoine_data_path = os.path.join(SCRIPT_DIR, 'antoine_mipace_format.csv')
    
    # Prédire VO2max pour les données d'Antoine avec le modèle sans avgRE et eSV
    print("\nPrédiction VO2max pour Antoine (modèle sans avgRE et eSV):")
    predict_vo2max(antoine_data_path, features, scaler, svm_model, mlp_model)
    
    # Vérifier si les modèles avec avgRE et eSV existent déjà
    svm_model_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'svm_model_with_avgRE_eSV.pkl')
    mlp_model_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'mlp_model_with_avgRE_eSV.pkl')
    scaler_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'scaler_with_avgRE_eSV.pkl')
    
    if os.path.exists(svm_model_with_avgRE_eSV_path) and os.path.exists(mlp_model_with_avgRE_eSV_path) and os.path.exists(scaler_with_avgRE_eSV_path):
        print("\nChargement des modèles existants (avec avgRE et eSV)...")
        scaler_with_avgRE_eSV = joblib.load(scaler_with_avgRE_eSV_path)
        svm_model_with_avgRE_eSV = joblib.load(svm_model_with_avgRE_eSV_path)
        mlp_model_with_avgRE_eSV = joblib.load(mlp_model_with_avgRE_eSV_path)
        
        # Liste des caractéristiques utilisées pour la prédiction (avec avgRE et eSV)
        # L'ordre des caractéristiques doit correspondre exactement à celui utilisé lors de l'entraînement
        features_with_avgRE_eSV = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                   'RH Humidity', 'avgHR last 30', 'MaxSpeed', 'gender_encoded', 'Grade_encoded', 
                   'avgRE', 'eSV']
        
        # Prédire VO2max pour les données d'Antoine avec le modèle avec avgRE et eSV
        print("\nPrédiction VO2max pour Antoine (modèle avec avgRE et eSV):")
        
        # Charger les données avec avgRE et eSV
        df_antoine_for_avgRE_eSV = pd.read_csv(antoine_data_path)
        df_antoine_for_avgRE_eSV['gender_encoded'] = df_antoine_for_avgRE_eSV['gender'].map({'Male': 0, 'Female': 1})
        df_antoine_for_avgRE_eSV['Grade_encoded'] = df_antoine_for_avgRE_eSV['Grade'].map({True: 1, False: 0})
        
        # Créer un DataFrame avec les caractéristiques nécessaires
        X_pred_with_avgRE_eSV = df_antoine_for_avgRE_eSV[features_with_avgRE_eSV].copy()
        
        print(f"Caractéristiques utilisées pour la prédiction avec avgRE et eSV: {features_with_avgRE_eSV}")
        
        # Standardiser les données
        X_pred_scaled_with_avgRE_eSV = scaler_with_avgRE_eSV.transform(X_pred_with_avgRE_eSV)
        
        # Prédictions avec les deux modèles
        print("Prédiction avec le modèle SVM (avec avgRE et eSV)...")
        svm_predictions_with_avgRE_eSV = svm_model_with_avgRE_eSV.predict(X_pred_scaled_with_avgRE_eSV)
        
        print("Prédiction avec le modèle MLP (avec avgRE et eSV)...")
        mlp_predictions_with_avgRE_eSV = mlp_model_with_avgRE_eSV.predict(X_pred_scaled_with_avgRE_eSV)
        
        # Moyenne des prédictions (ensemble)
        ensemble_predictions_with_avgRE_eSV = (svm_predictions_with_avgRE_eSV + mlp_predictions_with_avgRE_eSV) / 2
        
        # Ajouter les prédictions au DataFrame
        output_path_with_avgRE_eSV = os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions_including_avgRE_eSV.csv')
        df_antoine_for_avgRE_eSV['VO2max_SVM_with_avgRE_eSV'] = svm_predictions_with_avgRE_eSV
        df_antoine_for_avgRE_eSV['VO2max_MLP_with_avgRE_eSV'] = mlp_predictions_with_avgRE_eSV
        df_antoine_for_avgRE_eSV['VO2max_Ensemble_with_avgRE_eSV'] = ensemble_predictions_with_avgRE_eSV
        df_antoine_for_avgRE_eSV.to_csv(output_path_with_avgRE_eSV, index=False)
        
        # Afficher les résultats
        print("\nRésultats des prédictions de VO2max (avec avgRE et eSV):")
        for i, row in df_antoine_for_avgRE_eSV.iterrows():
            print(f"ID: {row['id']}")
            print(f"  - VO2max (SVM avec avgRE et eSV): {row['VO2max_SVM_with_avgRE_eSV']:.2f} ml/kg/min")
            print(f"  - VO2max (MLP avec avgRE et eSV): {row['VO2max_MLP_with_avgRE_eSV']:.2f} ml/kg/min")
            print(f"  - VO2max (Ensemble avec avgRE et eSV): {row['VO2max_Ensemble_with_avgRE_eSV']:.2f} ml/kg/min")
        
        print(f"\nRésultats avec avgRE et eSV sauvegardés dans {output_path_with_avgRE_eSV}")
        
        # Créer un graphique comparatif des modèles avec et sans avgRE/eSV
        plt.figure(figsize=(12, 8))
        
        # Chargement des prédictions sans avgRE/eSV
        df_sans_avgRE_eSV = pd.read_csv(os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions.csv'))
        
        # Préparation des données pour le graphique
        modeles = ['SVM sans avgRE/eSV', 'MLP sans avgRE/eSV', 'Ensemble sans avgRE/eSV',
                   'SVM avec avgRE/eSV', 'MLP avec avgRE/eSV', 'Ensemble avec avgRE/eSV']
        
        valeurs = [df_sans_avgRE_eSV['VO2max_SVM'].iloc[0],
                  df_sans_avgRE_eSV['VO2max_MLP'].iloc[0],
                  df_sans_avgRE_eSV['VO2max_Ensemble'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_SVM_with_avgRE_eSV'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_MLP_with_avgRE_eSV'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_Ensemble_with_avgRE_eSV'].iloc[0]]
        
        # Création des couleurs
        couleurs = ['#3498db', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#2ecc71']
        
        # Création du graphique
        bars = plt.bar(modeles, valeurs, color=couleurs)
        
        # Ajout des valeurs au-dessus des barres
        for bar, val in zip(bars, valeurs):
            plt.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.title('Comparaison des prédictions de VO2max avec et sans avgRE/eSV')
        plt.ylabel('VO2max prédit (ml/kg/min)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(valeurs) * 1.15)  # Ajouter de l'espace au-dessus des barres
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Sauvegarder le graphique
        comparison_path = os.path.join(RESULTS_DIR, 'vo2max_predictions_with_without_avgRE_eSV.png')
        plt.savefig(comparison_path)
        
        print(f"Graphique de comparaison sauvegardé dans {comparison_path}")
    else:
        print("\nLes modèles avec avgRE et eSV n'existent pas. Exécutez d'abord l'entraînement avec ces paramètres.")
