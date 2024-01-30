"""This module handle making, training and saving the RNN, alongside outputing the results by either 
printing the error or plotting them with matplotlibb"""
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.metrics import mean_squared_error
from data import get_xy_from_data

def make_rnn(hidden_units, dense_units, input_shape, activation):
    """
    This function builds an RNN using keras
    """
    model = build_rnn_structure(hidden_units, dense_units, input_shape, activation)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def build_rnn_structure(hidden_units: int, dense_units: int, input_shape: (int, int),
                        activation: list[str]):
    """
    This function builds the structure for the RNN
    
    Parameters
    ----------
    hidden_units : int
        The hidden units for the RNN
    dense_units : int
        The dense units for the rnn
    input_shape : (int, int)
        The models input shape
    activation : list[str]
        The models activation thing

    """
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    model.add(Dense(dense_units, activation=activation[1]))
    return model

def train_model_on_data(model: Sequential, training_data, time_steps = 12, epochs = 20):
    """
    This function trains a model on the provided training data

    Parameters
    ----------
    model : Sequential
        The model to train
    training_data : numpy.ndarray
        The data to train the model on
    time_steps : int
        The time steps between the data
    epochs : int
        The epoch count for training the model
    """
    x, y = get_xy_from_data(training_data, time_steps)

    model.fit(x, y, epochs=epochs, batch_size=1, verbose=2)


def print_error(train_y, test_y, train_predict, test_predict):
    """
    This function prints the RMSE accuracy of the data to the command line

    Parameters
    ----------
    train_y : numpy.ndarray
        The actual data the model was trained on.
    test_y : numpy.ndarray 
        The actual data the model was tested on.
    train_predict : numpy.ndarray 
        The RNNs predictions for the data it was trained on.
    test_predict : numpy.ndarray 
        The RNNs predictions for the data it was tested on.

    See Also
    --------
    plot_results: Plots the arguments to a matplotlib graph

    Examples
    --------
    >>> print_error(train_y, test_y, train_predict, test_predict)
    Train RMSE: 0.088 RMSE
    Test RMSE: 0.096 RMSE
    """
    print(f"Type of train_y: {type(train_y)}, Type of test_y: {type(test_y)}, \
          Type of train_predict: {type(train_predict)}, Type of test_predict: {type(test_predict)}")

    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(train_y, train_predict))
    test_rmse = math.sqrt(mean_squared_error(test_y, test_predict))
    # Print RMSE
    print(f'Train RMSE: {train_rmse} RMSE')
    print(f'Test RMSE: {test_rmse} RMSE')

def plot_results(train_y, test_y, train_predict, test_predict):
    """
    This function plots the data to a matplotlib graph

    Parameters
    ----------
    train_y : numpy.ndarray
        The actual data the model was trained on.
    test_y : numpy.ndarray 
        The actual data the model was tested on.
    train_predict : numpy.ndarray 
        The RNNs predictions for the data it was trained on.
    test_predict : numpy.ndarray 
        The RNNs predictions for the data it was tested on.

    Using these arguments, this function plots the data and returns the figure and lines it plotted
    """
    actual = np.append(train_y, test_y)
    predicted = np.append(train_predict, test_predict)
    rows = len(actual)
    fig = plt.figure(figsize=(15, 6), dpi=80)
    line1 = plt.plot(range(rows), actual)
    line2 = plt.plot(range(rows), predicted)
    plt.axvline(x=len(train_y), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    return fig, line1, line2

def write_to_csv(train_y, test_y, train_predict, test_predict):
    """
    This function writes the data and predicted data to a CSV file

    Parameters
    ----------
    train_y : numpy.ndarray
        The actual data the model was trained on.
    test_y : numpy.ndarray 
        The actual data the model was tested on.
    train_predict : numpy.ndarray 
        The RNNs predictions for the data it was trained on.
    test_predict : numpy.ndarray 
        The RNNs predictions for the data it was tested on.

    This function utilizes these arguments to create a csv file with the following structure
    For each line there is: "(x),(realydata),(predictedydata)"
    """
    data = np.append(train_y, test_y)
    predicted = np.append(train_predict, test_predict)
    rows = len(data)

    with open("./out.csv", "w", encoding="utf-8") as f:
        for i in range(rows):
            f.write(f"{i},{data[i]},{predicted[i]}\n")
