# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:07:06 2022

@author: User
"""
from sklearn.neural_network import MLPRegressor
import math
import pandas as pd
import numpy as np
import data_manipulation
from sklearn import metrics


def nn(data,percentage_training,gap,
       n_iterations,
       
       hidden_layer_sizes,
       activation,
       learning_rate_init,
       max_iter
       ):
    
    mlpr = MLPRegressor(hidden_layer_sizes = hidden_layer_sizes,
       activation = activation,
       learning_rate_init = learning_rate_init,
       max_iter = max_iter)
    
    mean_rmse = [0,0]
    # mean_rmse = 0
    
    for i in range(n_iterations):
        
        data_train, data_test = data_manipulation.split_data_gap(data,percentage_training,gap)
        
        data_train_in = data_train[data_train.columns[0:6]]
        data_train_out = data_train[data_train.columns[6:8]]
        
        data_test_in = data_test[data_test.columns[0:6]]
        data_test_out = data_test[data_test.columns[6:8]]
        
        mlpr.fit(data_train_in,data_train_out)
        
        pred = mlpr.predict(data_test_in)
        
        # rmset = metrics.mean_squared_error(data_test_out,pred,squared = True)
        rmse0 = metrics.mean_squared_error(data_test_out[data_test_out.columns[0]],pred[:,0],squared = True)
        rmse1 = metrics.mean_squared_error(data_test_out[data_test_out.columns[1]],pred[:,1],squared = True)
        
        mean_rmse = [mean_rmse[0] + rmse0, mean_rmse[1] + rmse1]
        
        print("rmse 0 = " + str(math.sqrt(rmse0)) + "   1 = " + str(math.sqrt(rmse1)))
        # mean_rmse+=rmset
        # print(math.sqrt(rmset))
    mean_rmse = [math.sqrt(mean_rmse[0]/(n_iterations)),math.sqrt(mean_rmse[1]/(n_iterations))]
    # mean_rmse = math.sqrt(mean_rmse/n_iterations)
    return mean_rmse

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
data.drop_duplicates(keep='first', inplace=True)
data = data.reset_index(drop=True)

rmse = nn(data = data,
       percentage_training = 0.7,
       gap = 5,
       n_iterations = 100,
       
       hidden_layer_sizes = (10,10),#(10,10),#(20,20,20),#(50,50,50),
       activation = 'relu',#'relu',
       learning_rate_init = 0.01,
       max_iter = 100000)

print(rmse)