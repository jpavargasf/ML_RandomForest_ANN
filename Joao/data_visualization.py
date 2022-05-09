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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_manipulation

def module(n):
    if(n<0):
        return -n
    return n



data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")

#separa as variáveis entre as de entrada (input) e saída (output) do robô
robot_input = data[data.columns[0:6]]

#primeira coluna sobre velocidade linear e segunda sobre velocidade angular
robot_output = data[data.columns[6:8]]


v = robot_output[robot_output.columns[1]]
v = v.to_numpy()
# v.sort()
# plt.stem(v)

#separar entre grupos de [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
# a = np.arange(0.05,1.05,0.1)
a = np.arange(-0.9,1.1,0.1)
#a = np.arange(-1,1.1,0.2)
#separar as classes entre treino e teste

# X_train, X_test, y_train, y_test = train_test_split(robot_input, , test_size = 0.3) #test_size = 0.3 70% training and 30% test



discretized_data,sqr_error,data_classes,number_of_elem = data_manipulation.dicretize_data(v,a)

discretized_data.sort()

plt.stem(discretized_data)

sqr_error.sort()

plt.stem(sqr_error)
plt.show()
print(sum(sqr_error)/len(sqr_error))

"""----------------------Discretização das velocidades---------------------"""

vlinear = robot_output[robot_output.columns[0]].to_numpy()
vangular = robot_output[robot_output.columns[1]].to_numpy()

disc_vlinear_range = np.arange(0.05,1.05,0.1)
disc_vangular_range = np.arange(-0.9,1.1,0.1)
#disc_vangular = np.arange(-1,1.1,0.2)

vlinear_disc, vlinear_se,vlinear_classes,vlinear_noelem = data_manipulation.dicretize_data(vlineear,disc_vlinear_range)
vangular_disc, vangular_se,vangular_classes,vangular_noelem = data_manipulation.dicretize_data(vangular,disc_vangular_range)



"""------------------------------------------------------------------------"""
"""---------------------velocidade linear x angular------------------------"""

vlinear = robot_output[robot_output.columns[0]].to_numpy()
vangular = robot_output[robot_output.columns[1]].to_numpy()


plt.plot(vlinear,vangular,'o',color='black')
plt.title("velocidade linear x velocidade angular")
plt.show()

plt.plot(vangular,vlinear,'o',color='black')
plt.title("velocidade angular x velocidade linear")
plt.show()

"""------------------------------------------------------------------------"""
