from __future__ import division
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# trainingData = np.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
# testData = np.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)

trainingData = pd.read_csv('MNIST_training.csv')
testData = pd.read_csv('MNIST_test.csv')

if __name__ == '__main__':

    b_est = [35.0, 0]
    learningRate = 1e-18

    y = trainingData.iloc[:, 0] # Remove the coloumns from trainingData.
    X = trainingData.iloc[0:, 0:]
