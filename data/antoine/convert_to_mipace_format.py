import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, 'activity.csv')
output_file = os.path.join(script_dir, 'antoine_mipace_format.csv')


df_activity = pd.read_csv(input_file, sep=';')


df_mipace = pd.DataFrame(columns=[
    'id', 'visit', 'age', 'HR Max', 'gender', 'weight', 'height', 'bmi', 
    'Amb Temp', 'RH Humidity', 'Grade', 'avgHR last 30', 'MaxSpeed'
])


for index, row in df_activity.iterrows():

    activity_id = row['activity_id']

    max_hr = 194

    try:
        heartrate_data = json.loads(row['heartrate_data']) if isinstance(row['heartrate_data'], str) else []
        if heartrate_data:
            last_30_values = heartrate_data[-30:] if len(heartrate_data) > 30 else heartrate_data
            avg_hr_last_30 = sum(last_30_values) / len(last_30_values)
        else:
            avg_hr_last_30 = row['average_heartrate'] 
    except (json.JSONDecodeError, TypeError):
        avg_hr_last_30 = row['average_heartrate']
    

    max_speed = row['max_speed']
    

    gender = "Male" 
    age = 22  
    weight = 81  
    height = 186  
    bmi = weight / ((height/100) ** 2)

    ambient_temp = 15  
    humidity = 50 
    

    grade = True if row['total_elevation_gain'] > 0 else False
    
    

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
    

    df_mipace = pd.concat([df_mipace, pd.DataFrame([new_row])], ignore_index=True)

df_mipace.index.name = None
df_mipace = df_mipace.reset_index()
df_mipace.to_csv(output_file, index=False)

print(f"Conversion terminée. Fichier '{output_file}' créé avec {len(df_mipace)} activités.")
