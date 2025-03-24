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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_save_models(use_avgRE_eSV=False):
    print("Chargement des données d'entraînement...")

    df_train = pd.read_csv(os.path.join(DATA_DIR, 'mipace_mlproject.csv'))
    
    df_train = df_train.dropna()

    base_features = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                     'RH Humidity', 'avgHR last 30', 'MaxSpeed']
    
    df_train['gender_encoded'] = df_train['gender'].map({'Male': 0, 'Female': 1})
    df_train['Grade_encoded'] = df_train['Grade'].map({True: 1, False: 0})
    

    base_features.extend(['gender_encoded', 'Grade_encoded'])
    
    if use_avgRE_eSV:
        print("Entraînement avec avgRE et eSV inclus")
        features = base_features + ['avgRE', 'eSV']
    else:
        print("Entraînement sans avgRE et eSV")
        features = base_features

    X = df_train[features]
    y = df_train['V02max']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scaler_file = 'scaler_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'scaler_without_avgRE_eSV.pkl'
    joblib.dump(scaler, os.path.join(RESULTS_DIR, scaler_file))
    
    print("Entraînement du modèle SVM...")
    svm_model = SVR(kernel='rbf', C=10, gamma='scale')
    svm_model.fit(X_train_scaled, y_train)
    
    svm_file = 'svm_model_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'svm_model_without_avgRE_eSV.pkl'
    joblib.dump(svm_model, os.path.join(RESULTS_DIR, svm_file))
    
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
    
    mlp_file = 'mlp_model_with_avgRE_eSV.pkl' if use_avgRE_eSV else 'mlp_model_without_avgRE_eSV.pkl'
    joblib.dump(mlp_model, os.path.join(RESULTS_DIR, mlp_file))
    
    print(f"Modèles entraînés et sauvegardés avec succès! (avec avgRE et eSV: {use_avgRE_eSV})")

    X_test_scaled = scaler.transform(X_test)
    svm_score = svm_model.score(X_test_scaled, y_test)
    mlp_score = mlp_model.score(X_test_scaled, y_test)
    
    print(f"Score SVM (R²): {svm_score:.4f}")
    print(f"Score MLP (R²): {mlp_score:.4f}")
    
    return features, scaler, svm_model, mlp_model

def predict_vo2max(data_path, features, scaler, svm_model, mlp_model):
    print(f"Prédiction de VO2max pour les données dans {data_path}...")
    
    df_antoine = pd.read_csv(data_path)
    
    df_antoine['gender_encoded'] = df_antoine['gender'].map({'Male': 0, 'Female': 1})
    df_antoine['Grade_encoded'] = df_antoine['Grade'].map({True: 1, False: 0})

    X_pred = df_antoine[features].copy()
    
    print(f"Caractéristiques utilisées pour la prédiction: {features}")

    X_pred_scaled = scaler.transform(X_pred)

    print("Prédiction avec le modèle SVM...")
    svm_predictions = svm_model.predict(X_pred_scaled)
    
    print("Prédiction avec le modèle MLP...")
    mlp_predictions = mlp_model.predict(X_pred_scaled)

    ensemble_predictions = (svm_predictions + mlp_predictions) / 2

    df_antoine['VO2max_SVM'] = svm_predictions
    df_antoine['VO2max_MLP'] = mlp_predictions
    df_antoine['VO2max_Ensemble'] = ensemble_predictions
    

    output_path = os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions.csv')
    df_antoine.to_csv(output_path, index=False)

    print("\nRésultats des prédictions de VO2max:")
    for i, row in df_antoine.iterrows():
        print(f"ID: {row['id']}")
        print(f"  - VO2max (SVM): {row['VO2max_SVM']:.2f} ml/kg/min")
        print(f"  - VO2max (MLP): {row['VO2max_MLP']:.2f} ml/kg/min")
        print(f"  - VO2max (Ensemble): {row['VO2max_Ensemble']:.2f} ml/kg/min")
        
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
    
    plt.savefig(os.path.join(RESULTS_DIR, 'vo2max_predictions_comparison.png'))
    
    print(f"\nRésultats sauvegardés dans {output_path}")
    print(f"Graphique sauvegardé dans {os.path.join(RESULTS_DIR, 'vo2max_predictions_comparison.png')}")
    
    return df_antoine

if __name__ == "__main__":

    svm_model_path = os.path.join(RESULTS_DIR, 'svm_model_without_avgRE_eSV.pkl')
    mlp_model_path = os.path.join(RESULTS_DIR, 'mlp_model_without_avgRE_eSV.pkl')
    scaler_path = os.path.join(RESULTS_DIR, 'scaler_without_avgRE_eSV.pkl')
    
    if os.path.exists(svm_model_path) and os.path.exists(mlp_model_path) and os.path.exists(scaler_path):
        print("Chargement des modèles existants (sans avgRE et eSV)...")
        scaler = joblib.load(scaler_path)
        svm_model = joblib.load(svm_model_path)
        mlp_model = joblib.load(mlp_model_path)

        features = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                   'RH Humidity', 'avgHR last 30', 'MaxSpeed',
                   'gender_encoded', 'Grade_encoded']
    else:
        print("Entraînement de nouveaux modèles (sans avgRE et eSV)...")
        features, scaler, svm_model, mlp_model = train_and_save_models(use_avgRE_eSV=False)
        
        print("\nPour comparaison, entraînement de modèles avec avgRE et eSV...")
        features_with_avgRE_eSV, _, _, _ = train_and_save_models(use_avgRE_eSV=True)
    
    antoine_data_path = os.path.join(SCRIPT_DIR, 'antoine_mipace_format.csv')
    
    print("\nPrédiction VO2max pour Antoine (modèle sans avgRE et eSV):")
    predict_vo2max(antoine_data_path, features, scaler, svm_model, mlp_model)
    
    svm_model_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'svm_model_with_avgRE_eSV.pkl')
    mlp_model_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'mlp_model_with_avgRE_eSV.pkl')
    scaler_with_avgRE_eSV_path = os.path.join(RESULTS_DIR, 'scaler_with_avgRE_eSV.pkl')
    
    if os.path.exists(svm_model_with_avgRE_eSV_path) and os.path.exists(mlp_model_with_avgRE_eSV_path) and os.path.exists(scaler_with_avgRE_eSV_path):
        print("\nChargement des modèles existants (avec avgRE et eSV)...")
        scaler_with_avgRE_eSV = joblib.load(scaler_with_avgRE_eSV_path)
        svm_model_with_avgRE_eSV = joblib.load(svm_model_with_avgRE_eSV_path)
        mlp_model_with_avgRE_eSV = joblib.load(mlp_model_with_avgRE_eSV_path)

        features_with_avgRE_eSV = ['age', 'HR Max', 'weight', 'height', 'bmi', 'Amb Temp', 
                   'RH Humidity', 'avgHR last 30', 'MaxSpeed', 'gender_encoded', 'Grade_encoded', 
                   'avgRE', 'eSV']
        
        print("\nPrédiction VO2max pour Antoine (modèle avec avgRE et eSV):")

        df_antoine_for_avgRE_eSV = pd.read_csv(antoine_data_path)
        df_antoine_for_avgRE_eSV['gender_encoded'] = df_antoine_for_avgRE_eSV['gender'].map({'Male': 0, 'Female': 1})
        df_antoine_for_avgRE_eSV['Grade_encoded'] = df_antoine_for_avgRE_eSV['Grade'].map({True: 1, False: 0})

        X_pred_with_avgRE_eSV = df_antoine_for_avgRE_eSV[features_with_avgRE_eSV].copy()
        
        print(f"Caractéristiques utilisées pour la prédiction avec avgRE et eSV: {features_with_avgRE_eSV}")

        X_pred_scaled_with_avgRE_eSV = scaler_with_avgRE_eSV.transform(X_pred_with_avgRE_eSV)

        print("Prédiction avec le modèle SVM (avec avgRE et eSV)...")
        svm_predictions_with_avgRE_eSV = svm_model_with_avgRE_eSV.predict(X_pred_scaled_with_avgRE_eSV)
        
        print("Prédiction avec le modèle MLP (avec avgRE et eSV)...")
        mlp_predictions_with_avgRE_eSV = mlp_model_with_avgRE_eSV.predict(X_pred_scaled_with_avgRE_eSV)
        ensemble_predictions_with_avgRE_eSV = (svm_predictions_with_avgRE_eSV + mlp_predictions_with_avgRE_eSV) / 2

        output_path_with_avgRE_eSV = os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions_including_avgRE_eSV.csv')
        df_antoine_for_avgRE_eSV['VO2max_SVM_with_avgRE_eSV'] = svm_predictions_with_avgRE_eSV
        df_antoine_for_avgRE_eSV['VO2max_MLP_with_avgRE_eSV'] = mlp_predictions_with_avgRE_eSV
        df_antoine_for_avgRE_eSV['VO2max_Ensemble_with_avgRE_eSV'] = ensemble_predictions_with_avgRE_eSV

        print("\nRésultats des prédictions de VO2max (avec avgRE et eSV):")
        for i, row in df_antoine_for_avgRE_eSV.iterrows():
            print(f"ID: {row['id']}")
            print(f"  - VO2max (SVM avec avgRE et eSV): {row['VO2max_SVM_with_avgRE_eSV']:.2f} ml/kg/min")
            print(f"  - VO2max (MLP avec avgRE et eSV): {row['VO2max_MLP_with_avgRE_eSV']:.2f} ml/kg/min")
            print(f"  - VO2max (Ensemble avec avgRE et eSV): {row['VO2max_Ensemble_with_avgRE_eSV']:.2f} ml/kg/min")
        
        print(f"\nRésultats avec avgRE et eSV sauvegardés dans {output_path_with_avgRE_eSV}")
        

        plt.figure(figsize=(12, 8))

        df_sans_avgRE_eSV = pd.read_csv(os.path.join(RESULTS_DIR, 'antoine_with_vo2max_predictions.csv'))

        modeles = ['SVM sans avgRE/eSV', 'MLP sans avgRE/eSV', 'Ensemble sans avgRE/eSV',
                   'SVM avec avgRE/eSV', 'MLP avec avgRE/eSV', 'Ensemble avec avgRE/eSV']
        
        valeurs = [df_sans_avgRE_eSV['VO2max_SVM'].iloc[0],
                  df_sans_avgRE_eSV['VO2max_MLP'].iloc[0],
                  df_sans_avgRE_eSV['VO2max_Ensemble'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_SVM_with_avgRE_eSV'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_MLP_with_avgRE_eSV'].iloc[0],
                  df_antoine_for_avgRE_eSV['VO2max_Ensemble_with_avgRE_eSV'].iloc[0]]

        couleurs = ['#3498db', '#3498db', '#3498db', '#2ecc71', '#2ecc71', '#2ecc71']
  
        bars = plt.bar(modeles, valeurs, color=couleurs)

        for bar, val in zip(bars, valeurs):
            plt.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}', 
                     ha='center', va='bottom', fontweight='bold')
        
        plt.title('Comparaison des prédictions de VO2max avec et sans avgRE/eSV')
        plt.ylabel('VO2max prédit (ml/kg/min)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(valeurs) * 1.15) 
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        comparison_path = os.path.join(RESULTS_DIR, 'vo2max_predictions_with_without_avgRE_eSV.png')
        plt.savefig(comparison_path)
        
        print(f"Graphique de comparaison sauvegardé dans {comparison_path}")
    else:
        print("\nLes modèles avec avgRE et eSV n'existent pas. Exécutez d'abord l'entraînement avec ces paramètres.")
