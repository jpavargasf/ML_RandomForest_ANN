import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Leio os datasets e retiro o que preciso
df= pd.read_csv('DATASET_MobileRobotNav.csv', sep=';')
#print(df.describe())
#print(df)

from sklearn.model_selection import train_test_split


labels = np.array(['Out_Vel_Linear(m/s)','Out_Vel_Angula(rad/s)']).T
features= df.drop(columns=['Out_Vel_Linear(m/s)','Out_Vel_Angula(rad/s)'], axis = 1)
#print(features)


#labels= df['Out_Vel_Linear(m/s)']
#y2= df['Out_Vel_Angula(rad/s)']

features_list = list(features.columns)
print(features_list)

features_train, features_test, labels_train, labels_test= train_test_split(features,labels,test_size= 0.3)

print('Training Features Shape:', features_train.shape)
print('Training Labels Shape:', labels_train.shape)
print('Testing Features Shape:', features_train.shape)
print('Testing Labels Shape:', labels_test.shape)

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

#rf= RandomForestClassifier()
rf = MultiOutputRegressor(RandomForestRegressor(n_estimators = 1000, oob_score=True,random_state=42))

rf.fit(features_train,labels_train)

predictions= rf.predict(features_test)
# Use the forest's predict method on the test data
predictions = rf.predict(features_test)
# Calculate the absolute errors
errors = abs(predictions - labels_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


#from sklearn.metrics import classification_report, confusion_matrix

importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


#Calculate error
mape = 100 * (errors / labels_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')