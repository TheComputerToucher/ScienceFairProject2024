from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def get_XY_from_data(data, time_steps):
    Y_index = np.arange(time_steps, len(data), time_steps)
    Y = data[Y_index]

    X_rows = len(Y)
    X_unshaped = data[range(time_steps*X_rows)]
    X = np.reshape(X_unshaped, (X_rows, time_steps, 1))

    return X, Y


def get_data(url: str, split_percent = 0.8):
    csv = read_csv(url, usecols=[1], engine="python")
    data = np.array(csv.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data).flatten()
    data_len = len(scaled_data)

    split_pos = int(data_len * split_percent)

    train_data = scaled_data[range(split_pos)]
    test_data = scaled_data[split_pos:]

    return train_data, test_data, scaled_data

