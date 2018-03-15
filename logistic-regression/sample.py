# logistic_regression_kfold.py
# python 2.7.14
# logistic-regression with ROC Curve
# MNIST classify between 6 and 8
# Shah Zafrani

from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def logistic_regression(x, y, steps, lr):
    w = initialize_weights(x.shape[1])
    costs = []
    for s in range(steps):
        gradient, error = gradient_descent(x, y, w)
        w = w + lr * gradient
        costs.append(error)
    return w, costs

def gradient_descent(x, y, w):
    # parts of this function are lifted from https://github.com/michelucci/Logistic-Regression-Explained/blob/master/MNIST%20with%20Logistic%20Regression%20from%20scratch.ipynb
    # array of sigmoid squashed probabilities
    predictions = predict(x, w)
    # number of training cases
    m = x.shape[1]
    # get error cost by subtracting predictions from ground truths
    error_cost = (predictions - y).transpose()

    gradient = -1.0 / m * np.dot(x.transpose(), error_cost)

    return gradient, sum(abs(error_cost))

def sigmoid(z):
    return (1.0/(1.0 + np.exp(-z)))

def predict(features, weights):
    # get probabilities and squash them with the sigmoid function
    return sigmoid(np.dot(features, weights))

def normalize(x):
    return x / 255.0

def rescale_data(y):
    # for i in range(len(y)):
    #     # if it's a 6 we call it a zero label
    #     if y[i] == 6:
    #         y[i] = 0
    #     # if it's an 8 we call it a 1
    #     else:
    #         y[i] = 1
    return (y - 6) / 2 # this is much more efficient

def initialize_weights(feature_shape):
    # initialize weight vector
    w = np.zeros(feature_shape)
    return w

def calculate_tpr_and_fpr(test_x, test_y, optmized_w):
    predictions = predict(test_x, optimized_w)
    # print(predictions)
    # print(test_x)
    tpr = 0
    fpr = 0
    tnr = 0
    fnr = 0
    num_positive = sum(test_y == 1)
    num_negative = len(test_y) - num_positive
    assert (len(predictions) == len(test_y))
    for i in range(len(test_y)):
        if (test_y[i] == 1):
            if(predictions[i] > threshold):
                tpr += 1
            else:
                fnr += 1
        if (test_y[i] == 0):
            if(predictions[i] <= threshold):
                tnr += 1
            else:
                fpr += 1
    tpr = float(tpr) / num_positive
    fpr = float(fpr) / num_negative
    return tpr, fpr

if __name__ == '__main__':

    # hyper-parameters
    num_folds = 10
    learning_rate = 1e-2
    threshold = 0.5
    gradient_descent_steps = 1000

    # setting up data
    mnist_data = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)

    kf = KFold(n_splits=num_folds)
    kf.get_n_splits(mnist_data)

    # initialize lists to hold fpr and tpr values to be plotted later
    falsePositiveRates = []
    truePositiveRates = []
    error_costs = []

    print("Logistic Regression on MNIST 6's and 8's. \nUsing K-Fold Cross-Validation with {} Folds \nLearning Rate: {}    |    Gradient Descent Steps: {}".format(num_folds, learning_rate, gradient_descent_steps))

    for train_index, test_index in kf.split(mnist_data):

        folded_mnist_training = mnist_data[train_index]
        folded_mnist_test = mnist_data[test_index]
        # get and rescale labels
        y_train = rescale_data(np.array(folded_mnist_training[:, 0]))
        y_test = rescale_data(np.array(folded_mnist_test[:, 0]))
        # normalize data
        x_train =  normalize(np.array(folded_mnist_training[:,1:]))
        x_test = normalize(np.array(folded_mnist_test[:, 1:]))

        optimized_w, costs = logistic_regression(x_train, y_train, gradient_descent_steps, learning_rate)
        error_costs.append(costs)
        tpr, fpr = calculate_tpr_and_fpr(x_test, y_test, optimized_w)
        truePositiveRates.append(tpr)
        falsePositiveRates.append(fpr)

    average_tpr = sum(truePositiveRates)/len(truePositiveRates)
    average_fpr = sum(falsePositiveRates)/len(falsePositiveRates)
    print("Average True Positive Rate: {} \nAverage False Positive Rate: {}".format(average_tpr, average_fpr))

    falsePositiveRates += [0,1]
    falsePositiveRates.sort()
    truePositiveRates += [0,1]
    truePositiveRates.sort()

    plt.figure(1)
    plt.title("Gradient Descent Convergence, fold: 0")
    plt.plot(error_costs[0])

    plt.figure(2)
    plt.title('Receiver Operating Characteristic')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.plot([0,falsePositiveRate,1], [0,truePositiveRate,1])
    plt.plot(falsePositiveRates, truePositiveRates)
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
