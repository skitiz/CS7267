from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt


def predict_class(training_data, w):
    z = np.dot(training_data, w)
    return 1.0 / (1.0 + np.exp(-z))


def gradient_descent(training_data, training_labels, learning_rate, gradient_steps):
    w = np.zeros(training_data.shape[1])
    costs = []
    for s in range(gradient_steps):
        predictions = predict_class(training_data, w)
        m = training_data.shape[1]
        error_cost = (predictions - training_labels).transpose()
        gradient = -1.0 / m * np.dot(training_data.transpose(), error_cost)
        w = w + learning_rate * gradient
        costs.append(sum(abs(error_cost)))
    return w, costs


def calculate_class(testing_data, testing_labels, weights, threshold):
    predictions = predict_class(testing_data, weights)
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    sample_a = sum(testing_labels == 8)
    sample_b = len(testing_labels) - sample_a
    for i in range(len(testing_labels)):
        if testing_labels[i] == '8':
            if predictions[i] > threshold:
                true_positive += 1
            else:
                false_negative += 1
        if testing_labels[i] == '6':
            if predictions[i] < '7':
                true_negative += 1
            else:
                false_positive += 1
    true_positive = true_positive / sample_a
    false_positive = false_positive / sample_b
    return true_positive, false_positive


def split_data(learning_rate, threshold, gradient_steps):
    kf = KFold(n_splits=10)
    kf.get_n_splits(dataset)
    true_positive_rates = []
    false_positive_rates = []
    for training, testing in kf.split(dataset):
        training_split = dataset[training]
        testing_split = dataset[testing]
        training_labels = np.array(training_split[:, 0])
        training_data = normalize(np.array(training_split[:, 1:]))
        testing_labels = np.array(testing_split[:, 0])
        testing_data = normalize(np.array(testing_split[:, 1:]))
        weights, costs = gradient_descent(training_data, training_labels, learning_rate, gradient_steps)
        true_positive, false_positive = calculate_class(testing_data, testing_labels, weights, threshold)
        true_positive_rates.append(true_positive)
        false_positive_rates.append(false_positive)
        # print(training_labels.shape, training_data.shape, testing_labels.shape, testing_data.shape)
    return true_positive_rates, false_positive_rates, costs


def normalize(x):
    return x / 255


dataset = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)


def main():
    learning_rate = 1e-2
    threshold = 7
    gradient_steps = 1000
    true_positive, false_positive, costs = split_data(learning_rate, threshold, gradient_steps)
    average_true = sum(true_positive) / len(true_positive)
    average_false = sum(false_positive) / len(false_positive)
    average_true += [0, 1]
    average_false += [0, 1]
    average_true.sort()
    average_false.sort()
    # # plt.plot(costs[0])
    # # plt.figure(2)
    # plt.plot(average_true, average_false)
    # plt.plot([0, 1],  [0, 1], 'r--')
    # plt.show()
    plt.plot(costs[0])
    plt.show()


if __name__ == '__main__':
    main()
