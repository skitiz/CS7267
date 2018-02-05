import numpy as np
from collections import Counter

def knn(training, test_data, k):
    distances = []
    for group in training:
       for features in training[0:len(group)]:
            distance = np.sqrt(np.sum(np.power(np.array(features) - np.array(test_data), 2)))
            distances.append([distance, group[0]])

    svotes =  sorted(distances[:k])
    for i in range(0, k):
        result = distances[i][0]

    return result

def main():
    training = np.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
    training = training.astype(float)
    test_data = np.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)
    test_data = test_data.astype(float)
    k = 5
    distances = []
    correct = 0
    total = 0
    for data in test_data:
        for group in data[0:len(data)]:
            vote = knn(training, group, k)
            if data[0] == vote:
                correct += 1
            total += 1

    print('Accuracy :' , correct/total)

if __name__ == '__main__':
    main()
