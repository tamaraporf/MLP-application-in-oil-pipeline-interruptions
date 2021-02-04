
#importando as bibliotecas e classes
import pandas as pd
import keras
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Sequential
import numpy as np
#import matplotlib.pyplot as plt
import scipy as sp

#leitura dos dataframes
base = pd.read_csv('dadoraiz.csv')

#separando os atributos
previsores = base.iloc[:, 2:134].values
localizacao = base.iloc[:, 0].values
espessura = base.iloc[:, 1].values

#camada_entrada = Input(shape=(132,))
#camada_oculta1=Dense(units = 66, activation = 'sigmoid')(camada_entrada)
#camada_oculta2=Dense(units = 66, activation = 'sigmoid')(camada_oculta1)
#camada_saida1=Dense(units=1, activation='linear')(camada_oculta2)
#camada_saida2=Dense(units=1, activation='linear')(camada_oculta2)

#regressor = Model(inputs= camada_entrada,
#                   outputs= [camada_saida1, camada_saida2])
#regressor.compile(optimazer = 'Adam', loss = 'mse',
#                  metrics = ['mean_square_error'])
#regressor.fit(previsores, [localizacao, espessura],
#              epochs = 5000, batch_size=100)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, espessura_treinamento, espessura_teste, localizacao_treinamento, localizacao_teste = train_test_split(previsores, espessura, localizacao, test_size = 0.25)

#inicio da RNA(criação da rede)
regressor = Sequential()
#aidcionando as camadas
#camada oculta : neuronios = 132(entrada)+ 1(saida)= 133/2~66
regressor.add(Dense(units = 66, activation= 'relu', input_dim = 132))
regressor.add(Dense(units = 66, activation= 'relu'))
#camada de saída1
regressor.add(Dense(units = 1, activation= 'linear'))
#camada de saida2
regressor.add(Dense(units = 1, activation= 'linear'))
#compilação
regressor.compile(optimizer = 'adam', loss= 'mean_absolute_error',
                  metrics = ['mean_absolute_error'])

#treinamento da RNA
regressor.fit(previsores_treinamento, [espessura_treinamento, localizacao_treinamento], 
              batch_size=1500, epochs = 50000)

previsto_teste = regressor.predict(previsores_teste)
previsto_treinamento = regressor.predict(previsores_treinamento)
#utilizando os dados de teste
resultado = regressor.evaluate(previsores_teste, [espessura_teste, localizacao_teste])

# métricas para conjunto de teste e treinamento
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2_teste = r2_score(espessura_teste , previsto_teste)
MAE_teste = mean_absolute_error(espessura_teste, previsto_teste)
MSE_teste = mean_squared_error(espessura_teste, previsto_teste)

r2_treinamento = r2_score(espessura_treinamento, previsto_treinamento)
MAE_treinamento = mean_absolute_error(espessura_treinamento, previsto_treinamento)
MSE_treinamento = mean_squared_error(espessura_treinamento, previsto_treinamento)


