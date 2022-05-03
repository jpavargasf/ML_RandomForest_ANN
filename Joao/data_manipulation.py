# -*- coding: utf-8 -*-
"""
Autor: João Paulo Vargas da Fonseca
Data: 01/05/2022
Trabalho desenvolvido para a disciplina de Sistemas Inteligentes do Curso
de Engenharia Eletrônica da Universidade Tecnológica Federal do Paraná

Comentários:
"""

#-----------------------------------------------------------------------------
"""------------------------------------------------------------------------"""

"""
dicretize_data: discretiza um vetor de dados em valores representados por outro
                vetor, sempre buscando o menor erro
                
Entradas:       data : lista de dados
                values:valores nos quais data será convertido
                
Saídas:         discretized_data: data discretizada
                sqr_error: lista de erro quadrático de cada elemento
                data_classes: data representado por classes ou indexes de values
                number_of_elem: número de elementos em cada classe ou index
"""
def dicretize_data(data,values):
    #valores discretizados
    discretized_data = [0]*data.size
    
    #dados separados em classes de acordo com values
    data_classes = [0]*data.size
    
    #erro quadrático
    sqr_error = [0]*data.size
    
    #número de elementos de cada classe
    number_of_elem = [0]*values.size
    
    
    for i in range(0,data.size):
        
        #index auxiliar que representa a classe atual
        class_index = 0
        
        discretized_data[i] = values[class_index]
        
        sqr_error[i] = (discretized_data[i]-data[i])**2
        
        for j in range(1,values.size):
            
            aux_sqr_error = (data[i] - values[j])**2
            
            if(aux_sqr_error < sqr_error[i]):
                sqr_error[i] = aux_sqr_error
                
                class_index = j
                
                discretized_data[i] = values[j]
                
        
        data_classes[i] = class_index
        number_of_elem[class_index] += 1
        
    return discretized_data,sqr_error,data_classes,number_of_elem
    

"""------------------------------------------------------------------------"""

"""
split_data_index: divide uma lista em duas, de maneira a deixar uma porcentagem
                  de cada classe (ou elementos iguais) em uma lista e o restante
                  da classe em outra
                
Entradas:       data : lista de dados enumerados de acordo com sua classe, 
                       devendo conter valores desde 0 ao número de classes - 1
                n_classes: número de classes ou valores distintos
                percentage: percentagem de elementos de cada classe que será 
                            retornado na primeira lista
                
Saídas:         data1: (lista de lista)
                    primeira parte dos dados separada, sendo a percentagem re-
                    ferente à ela. 
                    O primeiro index da lista refere-se ao número da classe e 
                    o segundo a qual index de data
                class_index: (lista de lista)
                    segunda parte dos dados separados, segue o mesmo esquema 
                    de data1, a não ser que a porcentagem de dados neste é 
                    1 - percentagem
"""

def split_data_index(data,n_classes,percentage):
    class_index = [[] for _ in range(n_classes)]
    
    #separa os elementos de cada classe
    for i in range(len(data)):
        class_index[data[i]].append(i)
    
    
    data1 = [[] for _ in range(n_classes)]
    
    import random
    #escolhe quais elementos de cada classe serão separados
    #itera classe
    for i in range(n_classes):
        
        n_elem_class = len(class_index[i])#poderia até ter contador ali em cima
        
        n_elem_data1 = int(n_elem_class*percentage)
        
        for j in range(n_elem_data1):
            randn = random.randint(0,n_elem_class-1)
            data1[i].append(class_index[i][randn])
            n_elem_class -= 1
            class_index[i].pop(randn)
            
    
    
    #class_index se tornou data2
    return data1,class_index    

"""------------------------------------------------------------------------"""

"""
split_data:     divide um dataframe e uma lista em duas, buscando sempre colocar
                uma porcentagem dos dados de cada classe em uma primeira parte,
                e retorna dois dataframes e duas listas.
                A classe de cada elemento é considerada pela sua saída ou 
                data_output, ou seja, mesmo index do dataframe e da lista
                
Entradas:       data_input: (dataframe)
                    dataframe de dados que serão separados de acordo com 
                    data_output
                data_output: (lista)
                    lista de classes (ou inteiros) que deve conter valores de
                    0 ao número de classes - 1
                n_classes: (inteiro)
                    número de classes ou de únicos possíveis valores em
                    data_output
                percentage: (float)
                    percentagem dos dados de cada classe que ficarão em uma 
                    primeira parcela dividida dos dados
                
Saídas:         d1_in: (dataframe)
                    dataframe dos dados de entrada (data_input) contendo uma 
                    porcentagem, indicada por percentage, de cada classe, 
                    indicada por data_output
                d1_out: (lista)
                    lista contendo a primeira parte de data_output
                d2_in: (dataframe)
                    dataframe contendo a segunda parde de data_input
                d2_out: (lista)
                    lista contendo a segunda parte de data_output
"""

def split_data(data_input,data_output,n_classes,percentage):
    index1,index2 = split_data_index(data_output,n_classes,percentage)
    
    
    d1_in = [[] for _ in range(0)]
    d1_out = []
    d2_in = [[] for _ in range(0)]
    d2_out = []
    for i in range(n_classes):
        for j in range(len(index1[i])):
            d1_in.append(data_input[index1[i][j]:index1[i][j]+1])
            d1_out.append(data_output[index1[i][j]])
        
        for j in range(len(index2[i])):
            d2_in.append(data_input[index2[i][j]:index2[i][j]+1])
            d2_out.append(data_output[index2[i][j]])
    
    import pandas as pd
    d1_in = pd.concat(d1_in)
    d2_in = pd.concat(d2_in)
    return d1_in,d1_out,d2_in,d2_out
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        