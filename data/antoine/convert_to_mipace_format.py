#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Chemins des fichiers
script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'activity.csv')
output_file = os.path.join(script_dir, 'antoine_mipace_format.csv')

# Charger les données d'activité
df_activity = pd.read_csv(input_file, sep=';')

# Création d'un DataFrame vide pour le format mipace
df_mipace = pd.DataFrame(columns=[
    'id', 'visit', 'age', 'HR Max', 'gender', 'weight', 'height', 'bmi', 
    'Amb Temp', 'RH Humidity', 'Grade', 'avgHR last 30', 'MaxSpeed'
])

# Traitement pour chaque activité
for index, row in df_activity.iterrows():
    # Récupérer les données d'activité
    activity_id = row['activity_id']
    
    # Extraire les données
    max_hr = 194
    
    # Calculer la moyenne des dernières 30 minutes du rythme cardiaque si disponible
    try:
        heartrate_data = json.loads(row['heartrate_data']) if isinstance(row['heartrate_data'], str) else []
        if heartrate_data:
            # Calculer la moyenne des 30 dernières valeurs (ou moins si pas assez de données)
            last_30_values = heartrate_data[-30:] if len(heartrate_data) > 30 else heartrate_data
            avg_hr_last_30 = sum(last_30_values) / len(last_30_values)
        else:
            avg_hr_last_30 = row['average_heartrate']  # Utiliser la moyenne globale si pas de données détaillées
    except (json.JSONDecodeError, TypeError):
        avg_hr_last_30 = row['average_heartrate']
    
    # Vitesse maximale
    max_speed = row['max_speed']
    
    # Données personnelles fictives basées sur l'activité
    gender = "Male"  # À modifier si nécessaire
    age = 22  # À modifier si nécessaire
    weight = 81  # À modifier si nécessaire
    height = 186  # À modifier si nécessaire
    bmi = weight / ((height/100) ** 2)
    
    # Données environnementales moyennes
    ambient_temp = 15  # Température ambiante (°C)
    humidity = 50  # Humidité relative (%)
    
    # Information sur la pente (Grade)
    # True si elevation_gain > 0, sinon False
    grade = True if row['total_elevation_gain'] > 0 else False
    
    
    # Créer une entrée pour ce fichier
    new_row = {
        'id': f"ANT{str(activity_id)[-8:]}",
        'visit': 1,
        'age': age,
        'HR Max': max_hr,
        'gender': gender,
        'weight': weight,
        'height': height,
        'bmi': round(bmi, 1),
        'Amb Temp': ambient_temp,
        'RH Humidity': humidity,
        'Grade': grade,
        'avgHR last 30': avg_hr_last_30,
        'MaxSpeed': max_speed,
    }
    
    # Ajouter au DataFrame
    df_mipace = pd.concat([df_mipace, pd.DataFrame([new_row])], ignore_index=True)

# Ajouter la colonne d'index numérique au début
df_mipace.index.name = None
df_mipace = df_mipace.reset_index()

# Enregistrer dans un fichier CSV au format mipace
df_mipace.to_csv(output_file, index=False)

print(f"Conversion terminée. Fichier '{output_file}' créé avec {len(df_mipace)} activités.")
