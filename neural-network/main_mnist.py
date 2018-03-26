'''
Basic code structure referenced from : https://github.com/DouglasGray/MNIST/blob/master/main_MNIST.py
'''

import NN
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import mnist_loader

# def load_data():
#     train, val, test = mnist_loader.load_data_wrapper()
#     train = list(train)
#     val = list(val)
#     test = list(test)
#
#     return train, val, test

# Wrote a custom load_data function for this cause I can't use MNIST_loader()
def load_data():
    # Reference : http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    data = pd.read_csv('MNIST_CV.csv', delimiter = ',', skiprows = 1, dtype = int)

    X = np.array(data[:, 1:])
    y = np.array(data[:, 0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    return (X_train, y_train), (X_test, y_test)



if __name__ == "__main__":
    training_data, validation_data, test_data = load_data()

    # Set network parameters and cost/transfer functions
    layers = [784, 30, 10]
    lmbda = 5.0
    eta = 0.5
    epochs = 30
    nbatch = 10

    cost = NN.CrossEntropy
    transfer_fns = [NN.sigmoid, NN.sigmoid]

    # Create the network, train and test
    nn = NN.Network(layers, cost, transfer_fns)
    nn.train_network(training_data, validation_data, epochs, nbatch, eta, lmbda)

    acc = nn.test_network(test_data)
    print("Accuracy on test data: {} / {})".format(acc, len(test_data)))
