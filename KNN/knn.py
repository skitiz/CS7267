"""
CS7267
Kshitij Bantupalli
Homework 1


Import the packages for the program.
"""

from __future__ import division
import numpy as np
from collections import Counter

'''
Main function.
'''


def main():
    """
    Import training and test data from MNIST.
    """
    training = np.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
    test_data = np.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)

    '''
    Declare the variables for the code.
    '''
    # Using k = 7. Modify this to change the accuracy.
    k = 7

    neighbors = np.zeros(10, dtype=int)
    prediction = []
    groundtruth = []
    distances = []
    correct = 0
    total = 0
    distance = 0
    wrong = 0
    accuracy = 0
    counter = 0

    # Calculate K nearest neighbors.

    for data in test_data:
        distances = []
        for row in training:
            distance = 0
            for x, y in zip(row[0:len(row)], data):
                distance += (x - y) ** 2
            temp = (np.sqrt(distance), int(row[0]))
            distances.append(temp)

        distances = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(distances).most_common(1)[0][0]
        # prediction.append(vote_result)
        # groundtruth.append(data[0])
        if vote_result == data[0]:
            correct += 1
        else:
            wrong += 1
    # 'Type Error: Bool object is not iterable' (?)
    # accuracy = sum(prediction == groundtruth) / len(groundtruth)
    accuracy = correct / (correct + wrong)
    print(accuracy * 100)


if __name__ == '__main__':
    main()
