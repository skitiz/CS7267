import numpy as np


class MeanSquaredError:
    def __init__(self):
        pass

    @staticmethod
    def nabla(a, y):
        return a - y

    @staticmethod
    def call(a, y):
        return 0.5 * np.mean(np.linalg.norm(y - a, axis=1) ** 2)


class Sigmoid:
    @staticmethod
    def call(z):
        return 1.0 / (1.0 + np.exp(-z))

    def prime(self, z):
        return self.call(z) * (1 - self.call(z))


class Softmax:
    @staticmethod
    def call(z):
        c = np.max(z)
        return np.exp(z - c) / np.sum(np.exp(z - c))


class Layer:
    def activation_prime(self):
        return self.activation_function.prime(self.weighted_input)

    def update_nabla(self, prev_layer):
        self.nabla(prev_layer)

    def nabla(self, prev_layer):
        if self.param_num() == 0:
            return

    def update_params(self, c):
        pass


class Dense(Layer):
    def __init__(self, n, activation=Sigmoid()):
        self.n = n
        self.activation_function = activation
        self.weighted_input = 0
        self.activation = 0

    def build(self, input_num):
        assert type(input_num) != tuple

        self.weight = np.random.rand(self.n, input_num) / np.sqrt(input_num)
        self.bias = np.random.rand(self.n)

        self.weight_nabla = np.zeros_like(self.weight)
        self.bias_nabla = np.zeros_like(self.bias)

        self.input_num = input_num

    def call(self, x):
        weighted_input = np.dot(self.weight, x) + self.bias
        activation = self.activation_function.call(weighted_input)

        self.weighted_input = weighted_input
        self.activation = activation

        return activation

    def backpropogation(self, prev_layer):
        return np.dot(self.weight.T, self.error) * prev_layer.activation_prime()

    def nabla(self, prev_layer):
        w_nabla = np.array([prev_layer.activation] * self.n) * np.array([self.error] * self.input_num).T
        b_nabla = self.error

        return w_nabla, b_nabla

    def update_nabla(self, prev_layer):
        w_nabla, b_nabla = self.nabla(prev_layer)

        self.weight_nabla += w_nabla
        self.bias_nabla += b_nabla

    def update_params(self, c):
        self.weight += c * self.weight_nabla
        self.bias += c * self.bias_nabla

        self.weight_nabla *= 0.
        self.bias_nabla *= 0

    def output_num(self):
        return self.n

    def param_num(self):
        weight_num = self.weight.shape[0] * self.weight.shape[1]
        bias_num = self.bias.shape[0]

        return weight_num + bias_num


class Input(Layer):
    def __init__(self, shape):
        self.shape = shape
        self.weighted_input = 0
        self.activation = 0

    def build(self):
        pass

    def call(self, x):
        self.weighted_input = x
        self.activation = x
        return x

    def output_num(self):
        return self.shape


class Network:
    def __init__(self, layers, loss=MeanSquaredError()):
        self.layers = layers
        self.layer_nums = []
        self.loss_function = loss

        for i, x in enumerate(self.layers):
            if i - 1 >= 0:
                x.build(self.layer_nums[i - 1])

            self.layer_nums.append(x.output_num())

    def sgd(self, train_data, val_data, test_data, epoch, batch_size, learning_rate = 0.1):
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        for epoch in range(epoch):
            for i in range(len(X_train) // batch_size):
                example_size = len(X_train)
                batch_index = np.random.choice(np.arange(example_size), size=batch_size)
                X_batch = X_train[batch_index]
                y_batch = X_train[batch_index]

                batch_num = len(X_batch)
                self.backpropogation(X_batch, y_batch)
                for layer in self.layers[1:]:
                    layer.update_params(-1 * learning_rate / batch_num)

            train_accuracy, train_loss= self.evaluate(X_batch, y_batch)
            valid_accuracy, valid_loss = self.evaluate(X_val, y_val)

            print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" % (
            epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

        print("Test accuracy L %f\ttest_loss: %f" % self.evaluate(X_test, y_test))

    def feedforward(self, x):
        activations = []

        if type(self.layer_nums[0]) == int:
            assert x.shape[0] == self.layer_nums[0]
        else:
            assert x.shape == tuple(self.layer_nums[0])

        times = []
        for l in range(len(self.layers)):
            if l - 1 >= 0:
                prev_activation = activations[l - 1]
            else:
                prev_activation = x

            activation = self.layers[l].call(prev_activation)
            activations.append(activation)

        return activations

    def backpropogation(self, X, y):
        batch_num = len(X)
        assert len(X) == len(y)

        for i, x in enumerate(X):
            activations = self.feedforward(x)

            last_layer = self.layers[-1]
            if self.loss_function.__class__.__name__ == 'MeanSquaredError':
                last_layer.error = last_layer.activation - y[i]
            else:
                nabla = self.loss_function.nabla(last_layer.activation, y[i])
                last_layer.error = nabla * last_layer.activation_function.prime(last_layer.weighted_input)

            for l in range(len(self.layers)-1, 1, -1):
                layer = self.layers[l]
                self.layers[l-1].error = layer.backpropogation(self.layers[l-1])

            for layer, prev_layer in zip(self.layers[1:], self.layers[0:-1]):
                layer.update_nabla(prev_layer)

    def evaluate(self, X, y):
        y_out = np.array([self.feedforward(x)[-1] for x in X])
        accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
        loss = self.loss_function.call(y_out, y)

        return accuracy, loss
