from __future__ import division
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasets.
trainingData = pd.read_csv('MNIST_training.csv', skiprows=[0], header=None)
testingData = pd.read_csv('MNIST_test.csv', skiprows=[0], header=None)

# Assigning the labels.
trainLabels = trainingData.iloc[:, 0]
testLabels = testingData[:, 0]

# Assigning the datasets.
trainData = trainingData[:, 1:]
# Normalize the training data with min max.
trainData = trainData / 255.0
testData = testingData[:, 1:]
# Normalize the test data with min max.
testData = testData / 255.0


def linearRegression(x, y):
    # Take from Dr. Kang's class presentation.
    return np.dot(np.dot(np.linalg.pinv(np.dot(x.transpose(), x)), x.transpose()), y)


if __name__ == '__main__':

    #
    # TASK 1 : CALCULATE B_OPT AND FIND ACCURACY OF PREDICTIONS VS TEST LABELS.
    #
    #
    # Calculate b_opt
    b_opt = linearRegression(trainData, trainLabels)

    # Printing the b_opt values.
    print(b_opt)

    # Calculating the predictions.
    predictions = np.array(np.dot(testData, b_opt, 0.5) > threshold)

    # Calculate the accuracy of the predictions.
    accuracy = sum(predictions == testLabels) / float(len(testLabels)) * 100

    print("accuracy of the model is :" accuracy)

    #
    # TASK 2 : CALCULATE THE B_ESTIMATE AND MINIMIZE THE COST FUNCTION.
    #

    m, n = trainData.shape
    b_est = np.zeros(p)
    
