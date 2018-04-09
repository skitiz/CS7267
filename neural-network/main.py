
import numpy as np
import pandas as pd
from neural_network import *
from sklearn.model_selection import train_test_split
np.random.seed(42)



def vectorize_array(x):
    return np.array([vectorize(y) for y in x])


def vectorize(x):
    v = np.zeros(10)
    v[x] = 1.0
    return v


def load():
    data = pd.read_csv('MNIST_CV.csv', delimiter=',', dtype=int, skiprows=1)

    X = np.array(data.iloc[:, 1:])
    y = np.array(data.iloc[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    y_train = vectorize_array(y_train)
    y_val = vectorize_array(y_val)
    y_test = vectorize_array(y_test)

    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    test_data = (X_test, y_test)

    return (train_data, val_data, test_data)

if __name__ == '__main__':

    epoch = 30
    batch_size = 30

    net = Network([Input(784),
                   Dense(30, activation=Sigmoid()),
                   Dense(10, activation=Softmax())], loss=MeanSquaredError())

    (train_data, val_data, test_data) = load()

    net.sgd(train_data, val_data, test_data, epoch, batch_size, learning_rate=0.1)
