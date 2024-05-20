import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Cargar datos de entrenamiento
train_data = pd.read_csv("./trainData.csv")
print(train_data.head())

# Cargar datos de prueba
test_data = pd.read_csv("./testData.csv")

# Añadir nueva característica ponderada
train_data['weightedGrade'] = train_data['averageGrade'] * (1 + 0.01 * train_data['totalAcciones'])
test_data['weightedGrade'] = test_data['averageGrade'] * (1 + 0.01 * test_data['totalAcciones'])

# Establecer la columna 'approved' basada en el nuevo umbral
train_data['approved'] = np.where(train_data['weightedGrade'] > 30, 'S', 'N')
test_data['approved'] = np.where(test_data['weightedGrade'] > 30, 'S', 'N')

# Separar características y etiquetas
y = train_data["approved"]
features = ["weightedGrade", "totalAcciones"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Entrenar el modelo
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# Predicciones
predictions = model.predict(X_test)

# Guardar resultados
output = pd.DataFrame({'Estudiante id': test_data.userid, 'Estudiante': test_data.firstname, 'Aprobó': predictions})
output.to_csv('submission.csv', index=False)


# Cargar datos de prueba
eval_data = pd.read_csv("./testDataEval.csv")
# Suponemos que test_data ya tiene una columna 'approved' con los valores reales.
y_eval = eval_data['approved']

# Calculamos el accuracy
accuracy = accuracy_score(y_eval, predictions) * 100

print(f"El porcentaje de acierto del modelo es: {accuracy:.2f}%")