#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:48:24 2020

@author: cesar
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 21:27:54 2020

@author: cesar
"""


# LSTM- RED NEURONAL RECURRENTES


#PARTE 1- Preprocesado de los datos

#importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importar data set de entrenamiento
path="/home/cesar/MIGTRA/tests/"
dataset_train = pd.read_csv(path+"daily_min_temperatures.csv")
training_set = dataset_train.iloc[:2555, 1:2].values  #hasta 1887, 70%

#Escalado de caracteristicas
#Para red neuronal recurrente es mejor utilizar la normalización MinMaxScaler, ya que converge más rapido
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled= sc.fit_transform(training_set)

#Crear una estructura de datos con 30 timesteps(30 dias) y 1 salida
x_train = []
y_train = []
timesteps= 30
for i in range(timesteps,training_set.shape[0]):
    x_train.append(training_set_scaled[i-timesteps:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train,y_train = np.array(x_train), np.array(y_train)

#Redimensión de datos
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


#PARTE 2- Construcción de los RNR
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model

#Inicialización del modelo
regressor = Sequential()

#Añadir la primera capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = True, input_shape=(x_train.shape[1],1) ))
regressor.add(Dropout(0.2))

#Añadir la 2da capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#Añadir la 3da capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = True ))
regressor.add(Dropout(0.2))

#Añadir la 4da capa de LSTM y la regularización por Dropout
regressor.add(LSTM(units = 50, return_sequences = False ))
regressor.add(Dropout(0.2))

#Añadir la capa de salida
regressor.add(Dense(units=1))

#Compilar la RNR
regressor.compile(optimizer="adam", loss="mean_squared_error")


#visualización de la red
plot_model(regressor, to_file='model.png', show_shapes=False, show_layer_names=True,
                rankdir='TB', expand_nested=False, dpi=56)

#Ajustar la RNR a nuestro conjunto de entrenamiento
history=regressor.fit(x_train,y_train, epochs= 50, batch_size = 32) #se sueleponer 100 epochs y 32 batches


#Guardar modelo y pesos
regressor.save("RNR_LSTM_temperature.h5")


#visualización de aprendizaje de red
loss = history.history['loss']
x = range(len(loss))
plt.figure(figsize=(12,5))
for key in history.history.keys():
    plt.plot(x, history.history[key], label=key)
plt.legend()


