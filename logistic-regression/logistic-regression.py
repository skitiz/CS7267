from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import time


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
    total_costs = []
    for training, testing in kf.split(dataset):
        training_split = dataset[training]
        testing_split = dataset[testing]
        training_labels = np.array(training_split[:, 0])
        training_data = normalize(np.array(training_split[:, 1:]))
        testing_labels = np.array(testing_split[:, 0])
        testing_data = normalize(np.array(testing_split[:, 1:]))
        weights, costs = gradient_descent(training_data, training_labels, learning_rate, gradient_steps)
        total_costs.append(costs)
        true_positive, false_positive = calculate_class(testing_data, testing_labels, weights, threshold)
        true_positive_rates.append(true_positive)
        false_positive_rates.append(false_positive)
        # print(training_labels.shape, training_data.shape, testing_labels.shape, testing_data.shape)
    average_true = sum(true_positive_rates) / len(true_positive_rates)
    average_false = sum(false_positive_rates) / len(false_positive_rates)
    average_true += [0, 1]
    average_false += [0, 1]
    print("Average True Positive Rate : ", average_true)
    print("Average False Positive Rate: ", average_false)
    average_true.sort()
    average_false.sort()
    plt.figure(1)
    plt.title("Gradient Descent Convergence, fold: 0")
    plt.plot(costs[0])

    plt.figure(2)
    plt.title('Receiver Operating Characteristic')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    # plt.plot([0,falsePositiveRate,1], [0,truePositiveRate,1])
    plt.plot(false_positive_rates, true_positive_rates)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    # return true_positive_rates, false_positive_rates, total_costs


def normalize(x):
    return x / 255


dataset = np.genfromtxt('MNIST_CV.csv', delimiter=',', dtype=int, skip_header=1)

start_time = time.time()


def main():
    learning_rate = 1e-2
    threshold = 7
    gradient_steps = 1000
    print("Logistic Regression on MNIST data. Classifying 6's and 8's.")
    split_data(learning_rate, threshold, gradient_steps)


if __name__ == '__main__':
    main()
