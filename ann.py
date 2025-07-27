# Instalar los requisitos para el programa
# pip3 install -r requirements.txt

import csv
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

VALOR = 'P20'
INDEX = ['P80', 'P50', 'P20'].index(VALOR) + 8

# Lectura de datos del archivo
with open("DATOS.csv") as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)

    data = []
    for row in reader:
        data.append({
            "parameters": [float(cell) for cell in row[:8]],
            "label": float(row[INDEX])
        })
data.sort(key = lambda item : item["label"])

# Normalizar los datos para mejorar el rendimiento
# Los datos originales fueron escalados al rango [0, 1] para ser procesados por
# la ANN, la cual usa la funcion sigmoid que devuelve valores en este rango
parameters = np.array([row["parameters"] for row in data])
labels = np.array([[row["label"]] for row in data])
scaler_parameters = MinMaxScaler(feature_range = (0, 1))
scaler_labels = MinMaxScaler(feature_range = (0, 1))
parameters_sc = scaler_parameters.fit_transform(parameters)
labels_sc = scaler_labels.fit_transform(labels)

# Separar los datos en grupos de entrenamiento y testing
X_training, X_testing, y_training, y_testing = train_test_split(
    parameters_sc, labels_sc, test_size=10
)

#################################################################
############## RED NEURONAL ARTIFICIAL (ANN) ####################
#################################################################

# Definir el modelo
model = tf.keras.models.Sequential()
# Capa intermedia con 13 nucleos y 8 nucleos de entrada, funcion de activacion sigmoid
model.add(tf.keras.layers.Dense(13, input_shape=(8,), activation='sigmoid'))
# Capa de salida, funcion de activacion sigmoid
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Compilar el modelo, algoritmo de optimizacion Momentum con ratio de aprendizaje 0.1
# y momentum 0.9, parametro para evaluar el rendimiento -> error cuadratico medio
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum = 0.9),
    loss='mean_squared_error',
    metrics=['RootMeanSquaredError']
)

# Entraner el modelo con 500 ciclos de aprendizaje
history = model.fit(X_training, y_training, epochs=500, validation_data=(X_testing, y_testing))

# Evaluar el rendimiento del modelo
model.evaluate(X_testing, y_testing, verbose=2)

# Graficar loss vs epochs, grafico de la mejora del rendimiento vs los ciclos de entrenamiento
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Loss en entrenamiento')
plt.plot(history.history['val_loss'], label='Loss en validación')
plt.xlabel('Epochs')
plt.ylabel('Loss (mean squared error)')
plt.title('Curva de entrenamiento')
plt.legend()
plt.grid(True)
plt.savefig('Loss_vs_Epochs.png')

#################################################################
############ REGRESION LINEAL MULTIPLE (MLR) ####################
#################################################################

# Definir el modelo
mlr = LinearRegression()

# Ajustar el modelo con los datos de entrenamiento
mlr.fit(X_training, y_training)

#################################################################
######### PREDICCION DE DATOS Y COMPARACION DE MODELOS ##########
#################################################################

################################################
###### DATOS DE ENTRENAMIENTO (TRAINING DATA)
################################################

    ### DATOS REALES
# Desnormalizar los datos reales
p80_real = scaler_labels.inverse_transform(y_training)

# Calculo de la media, desviacion estandar y coeficiente de variacion
media_real = np.mean(p80_real)
std_real = np.std(p80_real)
cv_real = std_real / media_real
print(f"Valores reales: Training")
print(f"Media: {media_real:.2f}")
print(f"Desviación estándar: {std_real:.2f}")
print(f"Coeficiente de variación: {cv_real:.2f}")

    ### MODELO ANN
p80_predicted_scaled = model.predict(X_training)
# Desnormalizar los datos obtenidos
p80_predicted_ANN = scaler_labels.inverse_transform(p80_predicted_scaled)

# Calculo de la media, desviacion estandar y coeficiente de variacion
media_ANN = np.mean(p80_predicted_ANN)
std_ANN = np.std(p80_predicted_ANN)
cv_ANN = std_ANN / media_ANN
print(f"Valores modelo ANN: Training")
print(f"Media: {media_ANN:.2f}")
print(f"Desviación estándar: {std_ANN:.2f}")
print(f"Coeficiente de variación: {cv_ANN:.2f}")
# Calculo del R²
r2_ANN = r2_score(p80_real, p80_predicted_ANN)
print(f"R² del modelo ANN: {r2_ANN:.4f}")

    ### MODELO MLR
p80_predicted_scaled = mlr.predict(X_training)
# Desnormalizar los datos obtenidos
p80_predicted_MLR = scaler_labels.inverse_transform(p80_predicted_scaled)

# Calculo de la media, desviacion estandar y coeficiente de variacion
media_MLR = np.mean(p80_predicted_MLR)
std_MLR = np.std(p80_predicted_MLR)
cv_MLR = std_MLR / media_MLR
print(f"Valores modelo MLR: Training")
print(f"Media: {media_MLR:.2f}")
print(f"Desviación estándar: {std_MLR:.2f}")
print(f"Coeficiente de variación: {cv_MLR:.2f}")
# Calculo del R²
r2_MLR = r2_score(p80_real, p80_predicted_MLR)
print(f"R² del modelo MLR: {r2_MLR:.4f}")

    ### GRAFICA - TRAINING DATA
# Ordenamiento de datos
p80_real = np.array(sorted(p80_real.tolist(), key=lambda item: item[0]))
p80_predicted_ANN = np.array(sorted(p80_predicted_ANN.tolist(), key=lambda item: item[0]))
p80_predicted_MLR = np.array(sorted(p80_predicted_MLR.tolist(), key=lambda item: item[0]))

plt.figure(figsize=(10,5))
plt.plot(p80_real, label=VALOR, marker='o')
plt.plot(p80_predicted_ANN, label=f"Modelo ANN - R² = {r2_ANN:.2f}", marker='x')
plt.plot(p80_predicted_MLR, label=f"Modelo MLR - R² = {r2_MLR:.2f}", marker='D')
plt.title(VALOR + ' vs ANN vs MLR (Training Data)')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.savefig('Comparacion_' + VALOR + '_training.png', dpi=300)
# plt.show()

################################################
###### DATOS DE PRUEBAS (TESTING DATA)
################################################

    ### DATOS REALES
# Desnormalizar los datos reales
p80_real = scaler_labels.inverse_transform(y_testing)
# Calculo de la media, desviacion estandar y coeficiente de variacion
media_real = np.mean(p80_real)
std_real = np.std(p80_real)
cv_real = std_real / media_real
print(f"Valores reales: Testing")
print(f"Media: {media_real:.2f}")
print(f"Desviación estándar: {std_real:.2f}")
print(f"Coeficiente de variación: {cv_real:.2f}")

    ### MODELO ANN
p80_predicted_scaled = model.predict(X_testing)
# Desnormalizar los datos obtenidos
p80_predicted_ANN = scaler_labels.inverse_transform(p80_predicted_scaled)

# Calculo de la media, desviacion estandar y coeficiente de variacion
media_ANN = np.mean(p80_predicted_ANN)
std_ANN = np.std(p80_predicted_ANN)
cv_ANN = std_ANN / media_ANN
print(f"Valores modelo ANN: Testing")
print(f"Media: {media_ANN:.2f}")
print(f"Desviación estándar: {std_ANN:.2f}")
print(f"Coeficiente de variación: {cv_ANN:.2f}")
# Calculo del R²
r2_ANN = r2_score(p80_real, p80_predicted_ANN)
print(f"R² del modelo ANN: {r2_ANN:.4f}")

    ### MODELO MLR
p80_predicted_scaled = mlr.predict(X_testing)
# Desnormalizar los datos obtenidos
p80_predicted_MLR = scaler_labels.inverse_transform(p80_predicted_scaled)

# Calculo de la media, desviacion estandar y coeficiente de variacion
media_MLR = np.mean(p80_predicted_MLR)
std_MLR = np.std(p80_predicted_MLR)
cv_MLR = std_MLR / media_MLR
print(f"Valores modelo MLR: Testing")
print(f"Media: {media_MLR:.2f}")
print(f"Desviación estándar: {std_MLR:.2f}")
print(f"Coeficiente de variación: {cv_MLR:.2f}")
# Calculo del R²
r2_MLR = r2_score(p80_real, p80_predicted_MLR)
print(f"R² del modelo MLR: {r2_MLR:.4f}")

    ### GRAFICA - TESTING DATA
# Ordenamiento de datos
p80_real = np.array(sorted(p80_real.tolist(), key=lambda item: item[0]))
p80_predicted_ANN = np.array(sorted(p80_predicted_ANN.tolist(), key=lambda item: item[0]))
p80_predicted_MLR = np.array(sorted(p80_predicted_MLR.tolist(), key=lambda item: item[0]))

plt.figure(figsize=(10,5))
plt.plot(p80_real, label=VALOR, marker='o')
plt.plot(p80_predicted_ANN, label=f"Modelo ANN - R² = {r2_ANN:.2f}", marker='x')
plt.plot(p80_predicted_MLR, label=f"Modelo MLR - R² = {r2_MLR:.2f}", marker='D')
plt.title(VALOR + ' vs ANN vs MLR (Testing Data)')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.savefig('Comparacion_' + VALOR + '_testing.png', dpi=300)
# plt.show()