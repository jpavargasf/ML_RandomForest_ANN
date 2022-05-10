# -*- coding: utf-8 -*-
"""
Created on Mon May  9 23:51:28 2022

@author: User
"""

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
from sklearn.model_selection import train_test_split


def nn(data,percentage_training,
       n_iterations,

       ):
    
    mlpr = MLPRegressor()
    
    mean_rmse = [0,0]
    # mean_rmse = 0
    
    for i in range(n_iterations):
        
        X = data[data.columns[0:6]]
        y = data[data.columns[6:8]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)   
        
        
        mlpr.fit(X_train,y_train)
        
        pred = mlpr.predict(X_test)
        
        # rmset = metrics.mean_squared_error(data_test_out,pred,squared = True)
        rmse0 = metrics.mean_squared_error(y_test[y_test.columns[0]],pred[:,0],squared = True)
        rmse1 = metrics.mean_squared_error(y_test[y_test.columns[1]],pred[:,1],squared = True)
        
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
       n_iterations = 100,
       
  )

print(rmse)