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
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

def rf_classifier(data,percentage_training,gap,
                  n_iterations,
                  discretization_values0,
                  discretization_values1,
                  
                  n_trees,
                  rf_max_depth,
                  rf_min_samples_split,
                  rf_max_categories,
                  rf_criterion
                  ):
    
    rf0 = RandomForestClassifier(n_estimators = n_trees,
                                criterion = rf_criterion,
                                min_samples_split=rf_min_samples_split,
                                max_features=rf_max_categories,
                                oob_score=True,
                                random_state=42
                                )
    
    rf1 = RandomForestClassifier(n_estimators = n_trees,
                                criterion = rf_criterion,
                                min_samples_split=rf_min_samples_split,
                                max_features=rf_max_categories,
                                oob_score=True,
                                random_state=42
                                )
    
    mean_rmse = [0,0]
    
    for i in range(n_iterations):
        
        data_train, data_test = data_manipulation.split_data_gap(data,percentage_training,gap)
        
        data_train_in = data_train[data_train.columns[0:6]]
        data_train_out = data_train[data_train.columns[6:8]]
        
        data_test_in = data_test[data_test.columns[0:6]]
        data_test_out = data_test[data_test.columns[6:8]]
        
        
        
        data_train_out0 = data_train_out[data_train_out.columns[0]].to_numpy()
        data_train_out1 = data_train_out[data_train_out.columns[1]].to_numpy()
        
        data_test_out0 = data_test_out[data_test_out.columns[0]].to_numpy()
        data_test_out1 = data_test_out[data_test_out.columns[1]].to_numpy()
        
        dto0 = data_manipulation.dicretize_data(data_train_out0,discretization_values0)[2]
        dto1 = data_manipulation.dicretize_data(data_train_out1,discretization_values1)[2]
        
        rf0.fit(data_train_in,dto0)
        rf1.fit(data_train_in,dto1)
        
        p0 = rf0.predict(data_test_in)
        p1 = rf1.predict(data_test_in)
        
        rv0 = [discretization_values0[j] for j in p0]
        rv1 = [discretization_values1[j] for j in p1]
        
        rmse0 = metrics.mean_squared_error(data_test_out0,rv0)
        rmse1 = metrics.mean_squared_error(data_test_out1,rv1)
        
        mean_rmse = [mean_rmse[0] + rmse0, mean_rmse[1] + rmse1]
        
        print("rmse 0 = " + str(math.sqrt(rmse0)) + "   1 = " + str(math.sqrt(rmse1)) + 
              "   oob 0 = " + str(rf0.oob_score_) + "  1 = "+str(rf1.oob_score_))
        
    mean_rmse = [math.sqrt(mean_rmse[0]/(n_iterations)),math.sqrt(mean_rmse[1]/(n_iterations))]
    return mean_rmse
        
        
        
def rf_regressor(data,percentage_training,gap,
                 n_iterations,
                 
                 n_trees,
                 rf_max_depth,
                 rf_min_samples_split,
                 rf_min_samples_leaf,
                 rf_min_impurity_decrease):
    
    rf0 = RandomForestRegressor(n_estimators = n_trees,
                               max_depth = rf_max_depth,
                               min_samples_split = rf_min_samples_split,
                               min_samples_leaf = rf_min_samples_leaf,
                               min_impurity_decrease = rf_min_impurity_decrease,
                               oob_score=True,
                               random_state=42)
    
    rf1 = RandomForestRegressor(n_estimators = n_trees,
                               max_depth = rf_max_depth,
                               min_samples_split = rf_min_samples_split,
                               min_samples_leaf = rf_min_samples_leaf,
                               min_impurity_decrease = rf_min_impurity_decrease,
                               oob_score=True,
                               random_state=42)
    mean_rmse = [0,0]
    # mean_abs_error = 0
    
    for i in range(n_iterations):
        
        data_train, data_test = data_manipulation.split_data_gap(data,percentage_training,gap)
        
        data_train_in = data_train[data_train.columns[0:6]]
        data_train_out = data_train[data_train.columns[6:8]]
        
        data_test_in = data_test[data_test.columns[0:6]]
        data_test_out = data_test[data_test.columns[6:8]]
        
        #velocidade linear
        rf0.fit(data_train_in,data_train_out[data_train_out.columns[0]])
        
        #velocidade angular
        rf1.fit(data_train_in,data_train_out[data_train_out.columns[1]])
        
        pred0 = rf0.predict(data_test_in)
        pred1 = rf1.predict(data_test_in)
        
        rmse0 = metrics.mean_squared_error(data_test_out[data_test_out.columns[0]],pred0,squared = True)
        rmse1 = metrics.mean_squared_error(data_test_out[data_test_out.columns[1]],pred1,squared = True)
        
        
        
        mean_rmse = [mean_rmse[0] + rmse0, mean_rmse[1] + rmse1]
        
        print("rmse 0 = " + str(math.sqrt(rmse0)) + "   1 = " + str(math.sqrt(rmse1)) + 
              "   oob 0 = " + str(rf0.oob_score_) + "  1 = "+str(rf1.oob_score_))
        
    mean_rmse = [math.sqrt(mean_rmse[0]/(n_iterations)),math.sqrt(mean_rmse[1]/(n_iterations))]
    return mean_rmse

data = pd.read_csv("DATASET_MobileRobotNav_FabroGustavo.csv")
data.drop_duplicates(keep='first', inplace=True)
data = data.reset_index(drop=True)

percentage_training = 0.7
gap = 5

rmse = rf_regressor(
    data = data,
    percentage_training = 0.7,
    gap = 5,
    n_iterations = 100,
    
    n_trees = 100,
    rf_max_depth = 5,
    rf_min_samples_split = 4,
    rf_min_samples_leaf = 2,
    rf_min_impurity_decrease = 0.0
    )
print(rmse)
data_train, data_test = data_manipulation.split_data_gap(data,percentage_training,gap)

rmse = rf_classifier(
    data = data,
    percentage_training = 0.7,
    gap = 5,
    n_iterations = 100,
    discretization_values0 = np.arange(0.05,1.05,0.1),
    discretization_values1 = np.arange(-0.9,1.1,0.1),
                  
    n_trees = 100,
    rf_max_depth = 3,
    rf_min_samples_split = 4,
    rf_max_categories = 3,
    rf_criterion = 'entropy'
    
    
    )
print(rmse)