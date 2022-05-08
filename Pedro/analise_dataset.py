import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#Leio os datasets e vejo uma descrição com média, 
df= pd.read_csv('DATASET_MobileRobotNav.csv', sep=';')
print(df.describe())

duplicados = df[df.duplicated(keep='first')]
print(duplicados)

df.drop_duplicates(keep='first', inplace=True)
print("-------")
print(df.describe())
# Criando o ambiente do gráfico 
sns.set_style("white")

#Ploto gráficos
plt.figure(figsize=(10, 10))
# Gráfico de Dispersão
g = sns.scatterplot(x="Modo", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Modo", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Frente", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Frente", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Esq30", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Esq30", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Esq45", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Esq45", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Dir30", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Dir30", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Dir45", y="Out_Vel_Linear(m/s)", 
                    data=df)
plt.show()

# Gráfico de Dispersão
g = sns.scatterplot(x="Sensor Dir45", y="Out_Vel_Angula(rad/s)", 
                    data=df)
plt.show()

#plota tabela de análise
fig, ax = plt.subplots()

# hide axes
describe = df.describe()
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')
ax.table(cellText=describe.values, colLabels=describe.columns, loc='center')

fig.tight_layout()

plt.show()