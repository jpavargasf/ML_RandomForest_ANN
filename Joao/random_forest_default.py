# -*- coding: utf-8 -*-
"""
Created on Mon May  9 23:16:27 2022

@author: User
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:27:23 2022

@author: User
"""

import math
import pandas as pd
import numpy as np
import data_manipulation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def rf_classifier(data,percentage_training,
                  n_iterations,
                  discretization_values0,
                  discretization_values1,
                  ):
    
    rf0 = RandomForestClassifier(
                                )
    
    rf1 = RandomForestClassifier(
                                )
    
    mean_rmse = [0,0]
    
    for i in range(n_iterations):
        X = data[data.columns[0:6]]
        y = data[data.columns[6:8]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)   
        
        yt0 = y_train[y_train.columns[0]].to_numpy()
        yte0 = y_test[y_test.columns[0]].to_numpy()
        
        yt1 = y_train[y_train.columns[1]].to_numpy()
        yte1 = y_test[y_test.columns[1]].to_numpy()
        
        
        
        

        
        dto0 = data_manipulation.dicretize_data(yt0,discretization_values0)[2]
        dto1 = data_manipulation.dicretize_data(yt1,discretization_values1)[2]
        
        rf0.fit(X_train,dto0)
        rf1.fit(X_train,dto1)
        
        p0 = rf0.predict(X_test)
        p1 = rf1.predict(X_test)
        
        rv0 = [discretization_values0[j] for j in p0]
        rv1 = [discretization_values1[j] for j in p1]
        
        rmse0 = metrics.mean_squared_error(yte0,rv0)
        rmse1 = metrics.mean_squared_error(yte1,rv1)
        
        mean_rmse = [mean_rmse[0] + rmse0, mean_rmse[1] + rmse1]
        
        print("rmse 0 = " + str(math.sqrt(rmse0)) + "   1 = " + str(math.sqrt(rmse1)))
        
    mean_rmse = [math.sqrt(mean_rmse[0]/(n_iterations)),math.sqrt(mean_rmse[1]/(n_iterations))]
    return mean_rmse
        
        
        
def rf_regressor(data,percentage_training,
                 n_iterations,
):
    
    rf0 = RandomForestRegressor()
    
    rf1 = RandomForestRegressor()
    mean_rmse = [0,0]
    # mean_abs_error = 0
    datain = data
    
    
    for i in range(n_iterations):
        X = data[data.columns[0:6]]
        y = data[data.columns[6:8]]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)   
        
        data_train_in = X_train
        data_train_out = y_train
        
        data_test_in = X_test
        data_test_out = y_test
        
        #velocidade linear
        rf0.fit(data_train_in,data_train_out[data_train_out.columns[0]])
        
        #velocidade angular
        rf1.fit(data_train_in,data_train_out[data_train_out.columns[1]])
        
        pred0 = rf0.predict(data_test_in)
        pred1 = rf1.predict(data_test_in)
        
        rmse0 = metrics.mean_squared_error(data_test_out[data_test_out.columns[0]],pred0,squared = True)
        rmse1 = metrics.mean_squared_error(data_test_out[data_test_out.columns[1]],pred1,squared = True)
        
        
        
        mean_rmse = [mean_rmse[0] + rmse0, mean_rmse[1] + rmse1]
        
        print("rmse 0 = " + str(math.sqrt(rmse0)) + "   1 = " + str(math.sqrt(rmse1)))
        
    mean_rmse = [math.sqrt(mean_rmse[0]/(n_iterations)),math.sqrt(mean_rmse[1]/(n_iterations))]
    return mean_rmse

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
# data.drop_duplicates(keep='first', inplace=True)
# data = data.reset_index(drop=True)

percentage_training = 0.7
gap = 5

rmse = rf_regressor(
    data = data,
    percentage_training = 0.7,
    n_iterations = 100
    )
print(rmse)
# data_train, data_test = data_manipulation.split_data_gap(data,percentage_training,gap)

rmse = rf_classifier(
    data = data,
    percentage_training = 0.7,
    n_iterations = 100,
    discretization_values0 = np.arange(0.05,1.05,0.1),
    discretization_values1 = np.arange(-0.9,1.1,0.1)
                  

    
    
    )
print(rmse)