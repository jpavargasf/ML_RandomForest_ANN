import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Leio os datasets e retiro o que preciso
df= pd.read_csv('DATASET_MobileRobotNav.csv', sep=';')
#Mostra a descrição do dataset (Média, Conta o número de tuplas, min, máx, etc)
print(df.describe())
#print(df)

#Separa o modelo para fazer a 
from sklearn.model_selection import train_test_split

#labels = np.array(df['Out_Vel_Linear(m/s)','Out_Vel_Angula(rad/s)'])
features= df.drop(columns=['Out_Vel_Linear(m/s)','Out_Vel_Angula(rad/s)'], axis = 1)
#print(features)


labels= df['Out_Vel_Linear(m/s)']
#y2= df['Out_Vel_Angula(rad/s)']

features_list = list(features.columns)
print(features_list)

features_train, features_test, labels_train, labels_test= train_test_split(features,labels,test_size= 0.3)

print('Training Features Shape:', features_train.shape)
print('Training Labels Shape:', labels_train.shape)
print('Testing Features Shape:', features_train.shape)
print('Testing Labels Shape:', labels_test.shape)

from sklearn.neural_network import MLPRegressor


mlp = MLPRegressor(hidden_layer_sizes=(8,8,8), activation='tanh', solver='adam', max_iter=500, alpha = 0.001)

mlp.fit(features_train,labels_train)

#Faz os testes separados anteriormente
predictions= mlp.predict(features_test)
print("a" ,predictions, "B:" ,labels_test)
#Calcula erros absolutos (mudar pra erro quadrático depois?)
errors = abs(predictions - labels_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Calcula qtd de erros
mape = 100 * (errors / labels_test)
#Mostra a precisão
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')