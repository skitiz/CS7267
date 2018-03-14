from sklearn import datasets,linear_model
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

kf = kFold(len())

// Perform k fold cross validation
// For each value of k train logistic regression using gradient descent
// Compute true positive rates and false positive rates for ROC curve
// Plot ROC curve for 10-cross validation

dataset =  pd.read_csv('MNIST_CV.csv', skiprows=[1], header=None)

if '__name__' == '__main__':
    main()