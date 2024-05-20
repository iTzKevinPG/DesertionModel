import numpy as np
import pandas as pd

# Cargar datos de entrenamiento
train_data = pd.read_csv("./trainData.csv")
print(train_data.head())

# Cargar datos de prueba
test_data = pd.read_csv("./testData.csv")
print(test_data.head())

# Calcular el número de aprobados
approved_count = train_data[train_data['approved'] == 'S'].shape[0]

# Calcular el porcentaje de aprobados
approved_percentage = (approved_count / len(train_data)) * 100

print(f"El porcentaje de aprobados es: {approved_percentage:.2f}%")
