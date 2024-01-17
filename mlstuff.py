from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.metrics import mean_squared_error
from data import get_XY_from_data
import math
import numpy as np
import matplotlib.pyplot as plt

def make_rnn(hidden_units, dense_units, input_shape, activation):
    model = build_rnn_structure(hidden_units, dense_units, input_shape, activation)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_rnn_structure(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation=activation[0]))
    model.add(Dense(dense_units, activation=activation[1]))
    return model

def train_model_on_data(model: Sequential, training_data, time_steps = 12, epochs = 20):
    x, y = get_XY_from_data(training_data, time_steps)

    model.fit(x, y, epochs=epochs, batch_size=1, verbose=2)


def print_error(trainY, testY, train_predict, test_predict):    
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    # Print RMSE
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))    

def plot_results(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predicted = np.append(train_predict, test_predict)
    rows = len(actual)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    line1 = plt.plot(range(rows), actual)
    line2 = plt.plot(range(rows), predicted)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    return fig, line1, line2

