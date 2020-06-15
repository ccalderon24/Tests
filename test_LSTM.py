#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:09:58 2020

@author: cesar
"""
#importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#importar data set
path="/home/cesar/MIGTRA/tests/"
dataset = pd.read_csv(path+"daily_min_temperatures.csv")
training_set = dataset.iloc[:2555, 1:2].values  #hasta 1887, 70%

#escalamos las variables de entrenamiento para luego escalar las de test
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled= sc.fit_transform(training_set)

timesteps=30
#PARTE 3- Ajustar las predicciones y visualizar los resultados
#Obtener el valor real de las temperaturas de 1888
dataset_test = dataset.iloc[2555:,:]
real_temperature = dataset_test.iloc[:, 1:2].values #desde 1888 en adelante, 30%

#Predecir las temperaturas con la RNR
dataset_total = dataset["Temp"]
inputs = dataset_total[len(dataset_total)-len(dataset_test)-timesteps:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

x_test = []
for i in range(timesteps,inputs.shape[0]):
    x_test.append(inputs[i-timesteps:i, 0])
x_test= np.array(x_test)

#Redimensión de datos
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

# Cargamos la LSTM
from keras.models import load_model
regressor = load_model('RNR_LSTM_temperature.h5')


predicted_temperature = regressor.predict(x_test)
predicted_temperature = sc.inverse_transform(predicted_temperature)



#Visualizar los Resultados
plt.plot(real_temperature,color='red', label="Temperatura Real ")
plt.plot(predicted_temperature,color='blue', label="Temperatura predicha")
plt.title("prediccion con una RNR del valor de las temperaturas en diferentes dias")
plt.xlabel("Fecha")
plt.ylabel("Temperatura")
plt.legend()
plt.savefig("prediction_tem_vs_real_temp.png")
plt.show()
