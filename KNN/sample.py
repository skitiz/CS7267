"""
Implementation of k-nearest neighbors on MNIST dataset.
"""

import numpy as np
from numpy import linalg as LA


'''
Returns array containing the k-nearest neighbors of point x
'''
def knn_x(data, x_data, k):
    n,m = data.shape
    # To calculate Euclidean distance efficiently,
    # use numpy subtraction/addition ops.
    # x_data = np.tile(x,(n,1))
    dist_matrix = LA.norm((x_data - data),axis=1)
    # Return an array containing the indexes k closest neighbors
    neighbors_indexes = np.argsort(dist_matrix)[:k]
    print(neighbors_indexes)
    return neighbors_indexes

'''
Returns single prediction based on the majority vote of neighbor labels
'''
def knn_clf_x(neighbors, true_labels):
    votes = []
    for idx in neighbors:
        votes.append(true_labels[idx])
    #print(votes)
    votes = np.array(votes,dtype ='int')
    #print(votes)
    return np.argmax(votes)

'''
K-nearest neighbors classifier implemented on test data.
'''
def knn_algorithm(train_data,train_labels,test_data,k):
	preds = []
	for x in test_data:
		neighbors = knn_x(train_data,x,k)
		preds.append(knn_clf_x(neighbors,train_labels))
	return np.array(preds)

'''
Returns unweighted errors of the classifiers
'''
def calc_error(true_labels,pred_labels):
    assert true_labels.size == pred_labels.size, "Vectors not equal size"
    n = true_labels.size
    correct = 0.0
    wrong = 0.0
    for i in range(n):
        if true_labels[i] == pred_labels[i]:
            correct += 1.0
        else:
            wrong += 1.0

    accuracy = correct/(correct+wrong)
    return accuracy

'''
Gets the training and validation error from the knn clf.
'''
def main():
    train = np.genfromtxt('MNIST_training.csv', delimiter=',', skip_header=1)
    test = np.genfromtxt('MNIST_test.csv', delimiter=',', skip_header=1)

    # Parse data for separating training labels and dataset
    n_feat = train[0].size
    print(n_feat)
    train_data = train[:,:-1]
    train_labels = train[:,n_feat-1]
    test_data = test[:,:-1]
    test_labels = test[:,n_feat-1]
    # Print training + validation error for the classifier
    k = 7
    preds_test = knn_algorithm(train_data,train_labels,test_data,k)
    result = calc_error(test_labels,preds_test)
    print(result)

if __name__ == '__main__':
	main()
