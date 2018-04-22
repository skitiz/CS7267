# Machine Learning (CS7267)

## Dependencies
* Python
* numpy
* Pandas
* MNIST Dataset
* sklearn

## Homework 1
# K - Nearest Neighbours
We use the MNIST handwritten letters dataset and use KNN to calculate the accuracy of prediction.
_Current accuracy : 90%_

*To run the code :* `python knn.py`

Modify the values of `k` inside to change accuracy levels.

Contributions are welcome!

## Homework 2
# Linear Regression
We use MNIST training and test data to calculate an optimal value for `b` to minimize the cost function.

*To run the code :* `python lr.py`

Modify the value for `learningRate` to try other possible answers.

## Homework 3
# Logisitc Regression
We use MNIST dataset and do 10-Fold Cross Validation to get training and test data. We find optimal weights and calculate the value of TPR and FPR to plot the ROC curve.

* To run the code :* `python logistic-regression.py`

Modify the value for `learning_rate`, `gradient_steps` and `splits` to tinker with it.

## Homework 4
# Neural Network
We used MNIST dataset to predict the hand written digit. The neural network has one hidden `Softmax` layer, the activation function of input layer is `Sigmoid()` and the the loss function is `Mean Squared Error`.

* To run the code :* `python main.py`

Modify the value for `learning_rate`, `epochs`, `batch_size` and `gradient_steps` to tinker with it.
