"""This module handles parsing the CSV data and extracting it to a format the RNN can read"""

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

def get_xy_from_data(data, time_steps):
    """
    Extracts the X and Y axis from the data csv
    """
    y_index = np.arange(time_steps, len(data), time_steps)
    y = data[y_index]

    x_rows = len(y)
    x_unshaped = data[range(time_steps*x_rows)]
    x = np.reshape(x_unshaped, (x_rows, time_steps, 1))

    return x, y


def get_data(url: str, split_percent = 0.8):
    """
    Reads a CSV from a path, splits and scales it 
    Returns the split data and the scaled data
    """
    csv = read_csv(url, usecols=[1], engine="python")
    data = np.array(csv.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data).flatten()
    data_len = len(scaled_data)

    split_pos = int(data_len * split_percent)

    train_data = scaled_data[range(split_pos)]
    test_data = scaled_data[split_pos:]

    return train_data, test_data, scaled_data
