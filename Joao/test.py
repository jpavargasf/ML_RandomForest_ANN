# -*- coding: utf-8 -*-
"""
Autor: João Paulo Vargas da Fonseca
Data: 30/04/2022
Trabalho desenvolvido para a disciplina de Sistemas Inteligentes do Curso
de Engenharia Eletrônica da Universidade Tecnológica Federal do Paraná

Comentários:
    "../DATASET_MobileRobotNav_FabroGustavo"
"""
#para o random forest, terá que separar entre classes
random_forest_test_size = 0.3


import pandas as pd

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")

import data_manipulation
a,b = data_manipulation.split_data_gap(data,0.7,5)
#separa as variáveis entre as de entrada (input) e saída (output) do robô
# robot_input = data[data.columns[0:6]]

# #primeira coluna sobre velocidade linear e segunda sobre velocidade angular
# #   Melhor separar ambas depois que for separado entre treinamento e teste, 
# #para utilizar o mesmo conjunto de teste para ambas saídas.
# robot_output = data[data.columns[6:8]]

# # robot_output_linear_v = data[data.columns[6]]
# # robot_output_angular_v = data[data.columns[7]]


# #separa as variáveis entre as de teste e treino
# from sklearn.model_selection import train_test_split
# robot_input_train,robot_input_test,robot_output_train,robot_output_test = train_test_split(robot_input,robot_output,test_size = random_forest_test_size)

# #random forest
# from sklearn.ensemble import RandomForestClassifier

# #parâmetros do random forest
# rf_n_trees = 100
# rf_n_max_categories = 3
# rf_criterion = "entropy"
# rf_max_samples = (robot_input_train.shape[0]-robot_input_train.shape[0]%rf_n_max_categories)/rf_n_max_categories
# rf_min_samples_split = 3
# #rf_n_jobs = -1#usar todos os processadores#n uso

# random_forest_linear_velocity = RandomForestClassifier(n_estimators = rf_n_trees,criterion = rf_criterion,min_samples_split=rf_min_samples_split,max_features=rf_n_max_categories,max_samples = rf_max_samples)

# random_forest_linear_velocity.fit(robot_input_train,robot_output_train[robot_output_train.columns[0]])
# #random_forest_linear_velocity.fit(robot_input_train,robot_output_train)

# robot_output_prediction = random_forest_linear_velocity.predict(robot_input_test[robot_input_test.columns[0]])