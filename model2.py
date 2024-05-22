import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Cargar datos de entrenamiento y prueba
train_data = pd.read_csv("./trainData.csv")
test_data = pd.read_csv("./testData.csv")

# Ponderación de las acciones: puedes ajustar estos pesos según lo que consideres adecuado
peso_ver = 1
peso_subir = 2

# Añadir características específicas y ponderadas
train_data['weightedActions'] = train_data['totalAccionesVer'] * peso_ver + train_data['totalAccionesSubir'] * peso_subir
test_data['weightedActions'] = test_data['totalAccionesVer'] * peso_ver + test_data['totalAccionesSubir'] * peso_subir

train_data['weightedGrade'] = train_data['averageGrade'] * (1 + 0.01 * train_data['weightedActions'])
test_data['weightedGrade'] = test_data['averageGrade'] * (1 + 0.01 * test_data['weightedActions'])

# Incorporar el número de cursos matriculados
train_data['coursesLoad'] = train_data['numCourses']
test_data['coursesLoad'] = test_data['numCourses']

# Establecer la columna 'approved' basada en el nuevo umbral
train_data['approved'] = np.where(train_data['weightedGrade'] > 3, 'S', 'N')
test_data['approved'] = np.where(test_data['weightedGrade'] > 3, 'S', 'N')

# Separar características y etiquetas
y = train_data["approved"]
features = ["weightedGrade", "weightedActions", "coursesLoad"]  # Actualizado para incluir las acciones ponderadas
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Entrenar el modelo con Gradient Boosting
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)
model.fit(X, y)

# Predicciones
predictions = model.predict(X_test)

# Guardar resultados
output = pd.DataFrame({'Estudiante id': test_data.userid, 'Estudiante': test_data.firstname, 'Aprobó': predictions})
output.to_csv('submission2.csv', index=False)

# Cargar datos de evaluación
eval_data = pd.read_csv("./testDataEval.csv")
y_eval = eval_data['approved']

# Calcular la precisión
accuracy = accuracy_score(y_eval, predictions) * 100

print(f"El porcentaje de acierto del modelo es: {accuracy:.2f}%")
