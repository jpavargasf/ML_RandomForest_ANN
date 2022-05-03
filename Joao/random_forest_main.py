# -*- coding: utf-8 -*-
"""
Autor: João Paulo Vargas da Fonseca
Data: 01/05/2022
Trabalho desenvolvido para a disciplina de Sistemas Inteligentes do Curso
de Engenharia Eletrônica da Universidade Tecnológica Federal do Paraná

Comentários:
"""

random_forest_test_size = 0.3


import pandas as pd

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
# data = data.loc[data[data.columns[5]]==1]

#separa as variáveis entre as de entrada (input) e saída (output) do robô
robot_input = data[data.columns[0:6]]

#primeira coluna sobre velocidade linear e segunda sobre velocidade angular
#   Melhor separar ambas depois que for separado entre treinamento e teste, 
#para utilizar o mesmo conjunto de teste para ambas saídas.
robot_output = data[data.columns[6:8]]

import numpy as np

v = robot_output[robot_output.columns[0]]
v = v.to_numpy()

#separar entre grupos de [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
discretization_values = np.arange(0.05,1.05,0.1)

#separar as classes entre treino e teste

# X_train, X_test, y_train, y_test = train_test_split(robot_input, , test_size = 0.3) #test_size = 0.3 70% training and 30% test

import data_manipulation

discretized_data,sqr_error,data_classes,number_of_elem = data_manipulation.dicretize_data(v,discretization_values)

# a1,a2 = data_manipulation.split_data_index(data_classes,10,0.3)
# from sklearn.model_selection import train_test_split

rin_train,rout_train,rin_test,rout_test = data_manipulation.split_data(robot_input,data_classes,discretization_values.size,0.7)

#comparar com a outra função já pronta
# from sklearn.model_selection import train_test_split
# rin_train,rin_test,rout_train,rout_test = train_test_split(robot_input,data_classes,test_size = 0.3)


from sklearn.ensemble import RandomForestClassifier

rf_n_trees = 100
rf_n_max_categories = 5
rf_criterion = "entropy"
rf_max_samples = None#int(rin_train.shape[0]/rf_n_max_categories)
rf_min_samples_split = 2#3

random_forest_linear_velocity = RandomForestClassifier(n_estimators = rf_n_trees,criterion = rf_criterion,min_samples_split=rf_min_samples_split,max_features=rf_n_max_categories,max_samples = rf_max_samples)

random_forest_linear_velocity.fit(rin_train,rout_train)

robot_output_prediction = random_forest_linear_velocity.predict(rin_test)

from sklearn import metrics
print(metrics.accuracy_score(rout_test, robot_output_prediction))
