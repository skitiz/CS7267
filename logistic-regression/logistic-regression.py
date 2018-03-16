# Importing packages for the program.
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import time


# Calculates the sigmoid and returns it.
def calculate_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Calculates the gradient and returns it.
def calculate_gradient(m, x):
    return -1.0 / m * x


# Calculates the logistic regression for training data and returns weights and optimized cost.
def logistic_regression(train_data, train_labels, learning_rate, gradient_steps):
    weights = np.zeros(train_data.shape[1])
    costs = []
    for steps in range(gradient_steps):
        sigmoid = calculate_sigmoid(np.dot(train_data, weights))
        m = train_data.shape[1]
        error_cost = (sigmoid - train_labels).transpose()
        gradient = calculate_gradient(m, np.dot(train_data.transpose(), error_cost))
        costs.append(sum(abs(error_cost)))
        weights = weights + learning_rate * gradient
    return weights, costs


# Normalizes training and test data.
def normalize(x):
    return x / 255.0


# Calculates the true positive and false positive rates.
def calculate_tpr_fpr(weights, test_data, test_label, threshold):
    predictions = calculate_sigmoid(np.dot(test_data, weights))
    tpr = 0
    fnr = 0
    fpr = 0
    tnr = 0

    label_a = sum(test_label == 8)
    label_b = len(test_label) - label_a
    for i in range(len(test_label)):
        if test_label[i] == 8:
            if predictions[i] > threshold:
                tpr += 1
            else:
                fnr += 1
        if test_label[i] == 6:
            if predictions[i] < threshold:
                tnr += 1
            else:
                fpr += 1
    tpr = float(tpr) / label_a
    fpr = float(fpr) / label_b
    return tpr, fpr


# Plotting function.
def plotter(cost, fpr, tpr):
    plt.figure(1)
    plt.title("Gradient Descent Covergence")
    plt.plot(cost)
    plt.figure(2)
    plt.title("ROC Curve")
    plt.ylabel('True Positive')
    plt.xlabel('False Positive')
    plt.plot(fpr, tpr)
    plt.show()


# Main function.
def main():
    dataset = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)
    learning_rate = 1e-4
    gradient_steps = 100
    threshold = 0.5
    kf = KFold(n_splits=10)
    kf.get_n_splits(dataset)
    weights = []
    total_fpr = []
    total_tpr = []
    total_costs = []
    for training, testing in kf.split(dataset):
        training_split = dataset[training]
        testing_split = dataset[testing]
        train_data = normalize(np.array(training_split[:, 1:]))
        test_data = normalize(np.array(testing_split[:, 1:]))
        train_label = np.array(training_split[:, 0])
        test_label = np.array(testing_split[:, 0])
        weights, costs = logistic_regression(train_data, train_label, learning_rate, gradient_steps)
        tpr, fpr = calculate_tpr_fpr(weights, test_data, test_label, threshold)
        total_costs.append(costs)
        total_fpr.append(tpr)
        total_tpr.append(fpr)
    average_tpr = sum(total_fpr) / len(total_fpr)
    average_fpr = sum(total_tpr) / len(total_tpr)
    total_fpr += [0, 1]
    total_fpr.sort()
    total_tpr += [0, 1]
    total_tpr.sort()
    total_fpr.sort()
    plotter(total_costs[0], total_fpr, total_tpr)


if __name__ == '__main__':
    main()
