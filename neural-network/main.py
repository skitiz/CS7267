# NOTE : Importing libraries.

import numpy as np
import pandas as pd
from neural_network import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# NOTE : Load the data and pre-processing
class LoadData:
    @staticmethod
    def batch(self, X, y, batch_size=100):
        example_size = len(X)
        batch_index = np.random.choice(np.arange(example_size), size=batch_size)
        return X[batch_index], y[batch_index]

    def vectorize(x):
        v = np.zeros(10)
        v[x] = 1.0
        return v


    def vectorize_array(self, y):
        return np.array([self.vectorize(x) for x in y])


    def load(self):
        data = pd.read_csv('MNIST_CV.csv', delimiter=',', dtype=int, skiprows=1)

        X = np.array(data.iloc[:, 1:])
        y = np.array(data.iloc[:, 0])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

        y_train = self.vectorize_array(self, y_train)
        y_val = self.vectorize_array(self, y_val)
        y_test = self.vectorize_array(self, y_test)

        train_data = (X_train, y_train)
        val_data = (X_val, y_val)
        test_data = (X_test, y_test)

        return train_data, val_data, test_data


def evaluate(net, X, y):
    y_out = net.output(X)
    accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
    loss = net.loss(X, y)

    return accuracy, loss


if __name__ == '__main__':
    accuracy = []
    epoch_count = []

    train_data, valid_data, test_data = LoadData.load(LoadData)

    X_train, y_train = train_data
    X_val, y_val = valid_data
    X_test, y_test = test_data

    # NOTE: HyperParameters
    batch_size = 30
    # learning_rate = 10 ** np.random.uniform(-6, 1)
    learning_rate = 0.012589482703246748
    epoch = 50
    early_stop_wait = 0
    prev_valid_loss = 100000.

    # NOTE: Define the layers.
    net = Network([
        Input(784),
        Dense(100, activation=Sigmoid()),
        Dense(10, activation=Softmax()),
    ], loss=MeanSquaredError())

    for epoch in range(50):
        for i in range(len(X_train) // batch_size):
            X_batch, y_batch = LoadData.batch(LoadData, X_train, y_train, batch_size=batch_size)
            net.train_on_batch(X_batch, y_batch, learning_rate= learning_rate)


        train_accuracy, train_loss = evaluate(net, X_batch, y_batch)
        valid_accuracy, valid_loss = evaluate(net, X_val, y_val)

        accuracy.append(train_accuracy)
        epoch_count.append(epoch)

        if valid_loss > prev_valid_loss:
            early_stop_wait += 1

            if early_stop_wait >= 2:
                break

        else:
            early_stop_wait = 0

        prev_valid_loss = valid_loss

        print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (
        epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

    print("test_accuracy: %f\ttest_loss: %f" % evaluate(net, X_test, y_test))
    print(learning_rate)
    plt.plot(accuracy, epoch_count)
    plt.show()

