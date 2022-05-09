# -*- coding: utf-8 -*-
"""
Autor: João Paulo Vargas da Fonseca
Data: 01/05/2022
Trabalho desenvolvido para a disciplina de Sistemas Inteligentes do Curso
de Engenharia Eletrônica da Universidade Tecnológica Federal do Paraná

Comentários:
"""
import math
import pandas as pd
import numpy as np
import data_manipulation
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


"""
rf_in: dataframe
    input dos  da árvore (xtrain + xtest)
rf_out: dataframe
    output da árvore (ytrain + ytest)
discretization_values: array de float
    classes nas quais rf_out deve se ajustar
train_size: float [0,1]
    % dados que irão ser de treino
rf_n_trees: inteiro
    número de árvores
rf_n_max_categories: inteiro
    número máximo de categorias por árvore
rf_criterion: string
    escolhe o tipo de árvore
    "gini"
    "entropy" ->ID3
rf_max_samples: inteiro ou None ou float
    máximo número de amostras por árvore
rf_min_samples_split: inteiro
    mínimo número de amostras em uma folha requerida para se dividir
n_iterations: inteiro
    número de loops que o programa rodará
d_form_a: float
    usado na fórmula de transformação de classe para real
    f(classe) = classe*d_form_a + d_form_b
d_form_b:
"""
def robot_rf(rf_in,rf_out, discretization_values, train_size, rf_n_trees,
             rf_n_max_categories,rf_criterion,rf_max_samples,
             rf_min_samples_split,n_iterations,d_form_a,d_form_b):
    
    rf_out_np = rf_out.to_numpy()
    
    discretized_data,sqr_error,data_classes,number_of_elem = data_manipulation.dicretize_data(
        rf_out_np,discretization_values)

    rf = RandomForestClassifier(n_estimators = rf_n_trees,
                                criterion = rf_criterion,
                                min_samples_split=rf_min_samples_split,
                                max_features=rf_n_max_categories,
                                max_samples = rf_max_samples)
    
    mean_accuracy = 0
    mean_rmse = 0
    
    for i in range(n_iterations):
        rin_train,rout_train,rin_test,rout_test = data_manipulation.split_data(
            rf_in,data_classes,discretization_values.size,train_size)
        
        #sobrescreve o fit
        rf.fit(rin_train,rout_train)
        
        rf_out_pred = rf.predict(rin_test)
        
        ma = metrics.accuracy_score(rout_test, rf_out_pred)
        
        mean_accuracy += ma
        
        #rmse = 0
        #calcula o rmse    
        
        d_rout_pred = [j*d_form_a + d_form_b for j in rf_out_pred]
        
        test_indexes = list(rin_test.index)
        real_rout_test = rf_out.iloc[test_indexes]
        #é true pq tem q vou tirar o quadrado depois
        rmse = metrics.mean_squared_error(real_rout_test,d_rout_pred,squared = True)
        mean_rmse += rmse
    
        print("Teste " + str(i) + "_Precisão = " + str(ma) + "_RMSE = " + str(math.sqrt(rmse)))
    
    mean_accuracy = mean_accuracy/n_iterations
    mean_rmse = math.sqrt(mean_rmse/n_iterations)
    
    return mean_accuracy,mean_rmse

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
data.drop_duplicates(keep='first', inplace=True)
data = data.reset_index(drop=True)
#se quiser separar 1 e -1, descomentar linha abaixo
#data = data.loc[data[data.columns[5]]==1]

#separa as variáveis entre as de entrada (input) e saída (output) do robô
robot_input = data[data.columns[0:6]]

#primeira coluna sobre velocidade linear e segunda sobre velocidade angular
#   Melhor separar ambas depois que for separado entre treinamento e teste, 
#para utilizar o mesmo conjunto de teste para ambas saídas.
robot_output = data[data.columns[6:8]]

#separar entre grupos de [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
discret_values = np.arange(0.05,1.05,0.1)

print("\nTeste de velocidade linear RANDOM FOREST\n")

accuracy_velocity,rmse_velocity = robot_rf(
    rf_in = robot_input,
    rf_out = robot_output[robot_output.columns[0]],
    discretization_values = discret_values,
    train_size = 0.7,
    rf_n_trees = 100,                     
    rf_n_max_categories = 5,
    rf_criterion = "entropy",
    rf_max_samples = None,
    rf_min_samples_split = 2,
    n_iterations = 10,
    d_form_a = 0.1,
    d_form_b = 0.05)

print("\nMédia Precisão = " + str(accuracy_velocity) + " RMSE = " + str(rmse_velocity))

"""-------------------------------------------------------------------------"""
""""necessita de ajustes para a velocidade angular (ou não, já achei o bug)"""
#10 classes - aumenta a precisão de classe, mas também aumenta rmse
# discret_values2 = np.arange(-1,1.1,0.2)
# df_a = 0.2
# df_b = -1

#20 classes
discret_values2 = np.arange(-0.9,1.1,0.1)
df_a = 0.1
df_b = -0.9

print("\nTeste de velocidade angular RANDOM FOREST\n")

accuracy_velocity,rmse_velocity = robot_rf(
    rf_in = robot_input,
    rf_out = robot_output[robot_output.columns[1]],
    discretization_values = discret_values2,
    train_size = 0.7,
    rf_n_trees = 100,                     
    rf_n_max_categories = 5,
    rf_criterion = "entropy",
    rf_max_samples = None,
    rf_min_samples_split = 2,
    n_iterations = 10,
    d_form_a = df_a,
    d_form_b = df_b)

print("\nMédia Precisão = " + str(accuracy_velocity) + " RMSE = " + str(rmse_velocity))