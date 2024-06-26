import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar datos
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
train_data['coursesLoad'] = train_data['numCourses']
test_data['coursesLoad'] = test_data['numCourses']

# Normalizar las características
scaler = StandardScaler()
features = ["weightedGrade", "weightedActions", "coursesLoad"]  # Actualizado para incluir las acciones ponderadas

X_train = train_data[features]
X_test = test_data[features]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Definir etiquetas
y_train = np.where(train_data['weightedGrade'] > 3, 'S', 'N')
y_test = np.where(test_data['weightedGrade'] > 3, 'S', 'N')

# Configuración de la búsqueda en cuadrícula
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_search = GridSearchCV(GradientBoostingClassifier(random_state=1), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Mejores parámetros y modelo
print(f"Mejores parámetros: {grid_search.best_params_}")
model = grid_search.best_estimator_

# Predicciones
predictions = model.predict(X_test_scaled)

# Guardar resultados
output = pd.DataFrame({'Estudiante id': test_data.userid, 'Estudiante': test_data.firstname, 'Aprobó': predictions})
output.to_csv('submission3.csv', index=False)

# Cargar datos de evaluación si están disponibles
eval_data = pd.read_csv("./testDataEval.csv")
y_eval = eval_data['approved']

# Calcular la precisión
accuracy = accuracy_score(y_eval, predictions) * 100
print(f"El porcentaje de acierto del modelo es: {accuracy:.2f}%")
