import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


base = pd.read_csv('dadoraiz.csv')

x = base.iloc[:, 2:135].values
y = base.iloc[:, [0,1]].values

#pré-processamento dos dados
#transformação dos atributos numéricos em atributos do tipo dummy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelenconder = LabelEncoder()
y[:, 0] = labelenconder.fit_transform(y[:, 0])

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0])],
                                  remainder='passthrough')

y = onehotencoder.fit_transform(y)

x= np.asarray(x).astype(np.float32)
y= np.asarray(y).astype(np.float32)

def criar_rede(loss, activation, neurons):
    regressor = Sequential()
    regressor.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = 'normal', input_dim = 133))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = 'normal'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 5, activation = 'linear'))
    regressor.compile(loss=loss, optimizer='adam', metrics=['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(build_fn = criar_rede,
                           epochs = 5000, batch_size = 100)
parametros = {'loss': ['mean_squared_error', 'mean_absolute_error', 'squared_hinge'],
              'activation': ['relu', 'sigmoid'],
              'neurons': [67, 34 ]}

grid_search = GridSearchCV(estimator = regressor, param_grid = parametros, cv = 5)

grid_search = grid_search.fit(x, y)
print (grid_search)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
