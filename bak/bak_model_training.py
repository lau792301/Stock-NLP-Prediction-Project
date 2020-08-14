# %%
import pandas as pd
import numpy as np

# %%
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense, UpSampling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed, Lambda, RepeatVector
from sklearn.model_selection import train_test_split

# %%
# TODO: read from csv
# Create Radom Data
'''
x1 0.0 0.2 0.4 0.6
x2  3   4    5  6
x3 100 200 -100 -200
x4 0.5 0.7 -0.1 -0.2
Yt+1= x1 * x2 +100 * x4
       
'''
length = 100
df = pd.DataFrame()
df['x1'] = [i/float(length) for i in range(length)]
df['x2'] = [i**2 for i in range(length)]
df['y'] = df['x1'] + df['x2'] 

df_value = df.values
x_value = df.drop(columns = 'y').values
y_value = df['y'].values.reshape(-1,1)
# %%


# class Model(object):
#     def __init__

'''
Need to DO
1. convert data can set input = n , output = n
2. split data (train, valid, test) and it can set shuffle with random seed
3. normalization data function
4. model
5. error_measurement by output (t_1: error, t_2: error)
'''
def build_data(x_value, y_value ,n_input, n_output):
    X, Y = list(), list()
    in_start = 0
    data_len = len(x_value)
    # step over the entire history one time step at a time
    for _ in range(data_len):
        # define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_output
        if out_end <= data_len:
            x_input = x_value[in_start:in_end] # e.g. t0-t3
            X.append(x_input)
            y_output = y_value[in_end:out_end] # e.g. t4-t5
            Y.append(y_output)
        # move along one time step
        in_start += 1
    return np.array(X), np.array(Y)            

X, Y = build_data(x_value, y_value, 3, 4)
# Y = Y.reshape(Y.shape[0], Y.shape[1])

# %%
def build_model(n_inputs, n_features, n_outputs, auto_encoder = True, n_repeat = 3):
    # define model
    model = Sequential()
    # LSTM AutoEncoder (Unsupervised Learning)
    # https://machinelearningmastery.com/lstm-autoencoders/
    model.add(LSTM(n_neurons, input_shape = (n_inputs, n_features), return_sequences=False))
    model.add(RepeatVector(n_repeat))
    model.add(LSTM(n_neurons, return_sequences=True))
    '''
    return_sequences default: False, return single hidden value (2 dimension)
                            True, return all time step hidden value. It also return lstm1, state_h, state_c.
                        
    '''
    # Dimension Adjustment
    model.add(UpSampling1D(n_outputs))
    model.add(Lambda(lambda x: x[:,-n_outputs:,:]))
    # https://stackoverflow.com/questions/55532683/why-is-timedistributed-not-needed-in-my-keras-lstm
    # model.add(Dense(1))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

def train_model(model, X, Y, shuffle = True, random_state = 1):
    train_x, valid_x, train_y, valid_y =  train_test_split(X, Y, shuffle = shuffle, random_state = random_state)
    history = model.fit(train_x, train_y, validation_data = (valid_x, valid_y))
    return model, history

n_neurons =  100
n_inputs, n_features  = X.shape[1], X.shape[2]
n_outputs = Y.shape[1]
n_repeat = 3

model = build_model(n_inputs = n_inputs,
                        n_features = n_features, n_outputs = n_outputs)
model, history = train_model(model , X, Y)




# %%
#  Shape Format: [samples, timesteps, features]
# define parameters
verbose, epochs, batch_size = 1, 20, 16
n_neurons =  100
n_inputs, n_features  = X.shape[1], X.shape[2]
n_outputs = Y.shape[1]

# RepeatVector
n_repeat = 3


# train_y = Y.reshape((Y.shape[0], Y.shape[1], 1))
# define model
model = Sequential()
# LSTM AutoEncoder (Unsupervised Learning)
# https://machinelearningmastery.com/lstm-autoencoders/
model.add(LSTM(n_neurons, input_shape = (n_inputs, n_features), return_sequences=False))
model.add(RepeatVector(n_repeat))
model.add(LSTM(n_neurons, return_sequences=True))
'''
return_sequences default: False, return single hidden value (2 dimension)
                          True, return all time step hidden value. It also return lstm1, state_h, state_c.
                    
'''
# Dimension Adjustment
model.add(UpSampling1D(n_outputs))
model.add(Lambda(lambda x: x[:,-n_outputs:,:]))
# https://stackoverflow.com/questions/55532683/why-is-timedistributed-not-needed-in-my-keras-lstm
# model.add(Dense(1))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=verbose)


'''
 The TimeDistributedDense layer allows you to build models that do the one-to-many and many-to-many architectures.
'''

