# -*- coding: utf-8 -*-
"""
Created on Sun May  1 02:06:01 2022

@author: User
"""

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


#separa as variáveis entre as de entrada (input) e saída (output) do robô
robot_input = data[data.columns[0:6]]

#primeira coluna sobre velocidade linear e segunda sobre velocidade angular
#   Melhor separar ambas depois que for separado entre treinamento e teste, 
#para utilizar o mesmo conjunto de teste para ambas saídas.
robot_output = data[data.columns[6:8]]

import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0,robot_output.shape[0])
v = robot_output[robot_output.columns[0]]
v = v.to_numpy()
v.sort()
plt.stem(v)