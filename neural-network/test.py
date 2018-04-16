class Network(object):
    def __init__ (self, layers, loss=MeanSquaredError()):

        # Initialize the layers with biases and define the loss function.
        self.layers = layers
        self.num_layers = []
        self.loss_function = loss

        for i,x in enumerate(self.layers):
            # Initialize biases and weights for all layers besides input layer.
            if i - 1 >= 0:
                layer.build(self.layer_nums[i-1])

            self.layer_nums.append(layer.output_num())

    def SGD(epoch, train_data, test_data, val_data, batch_size, learning_rate):

        # Split into labels and data
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data

        # Split into batches
        for epoch in range (epoch):
            for i in range(len(X_train) // batch_size):
                size = len(X_train)
                batch_index = np.random.choice(np.arange(size), size=batch_size)
                X_batch = X_train[batch_index]
                y_batch = y_train[batch_index]
                train_on_batch(X_batch, y_batch, learning_rate)

            train_accuracy, train_loss = evaluate(X_batch, y_batch)
            valid_accuracy, valid_loss = evaluate(X_valid, y_valid)

            print("epoch: %d\ttrain_accuracy: %f\ttrain_loss: %f\tvalid_accuracy: %f\tvalid_loss: %f" %
                  (epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))

            if valid_loss > prev_valid_loss:
                early_stop_wait += 1
            if early_stop_wait >= 2:
                break
            else:
                early_stop_wait = 0
            prev_valid_loss = valid_loss
        print("test_accuracy: %f\ttest_loss: %f" % evaluate(X_test, y_test))

    def train_on_batch(X, y, learning_rate):
        batch_num = len(X)
        assert len(X) == len(y)

        # Backpropogate
        self.backprop(X, y)

        # Update weights
        for layer in self.layers[1:]:
            layer.update_params(-1 * learning_rate / batch_num)

    def backprop(self, X, y):
        batch_num = len(X)
        assert len(X) == len(y)

        for i, x in enumerate(X):
            # forward pass
            activations = self.feedforward(x)

            # backward pass
            last_layer = self.layers[-1]
            if self.loss_function.__class__.__name__ == 'MeanSquaredError':
                last_layer.error = last_layer.activation = y[i]
            else:
                nabla = self.loss_function.nabla(last_layer.activation, y[i])
                last_layer.error = nabla * last_layer.activation_function.prime(last_layer.weighted_input)

            for l in range(len(self.layers) - 1, 1, -1):
                layer = self.layers[l]
                self.layers[l-1].error = layer.backprop(self.layers[l-1])

            for layer, prev_layer in zip(self.layers[1:], self.layers[0:-1]):
                layer.update_nabla(prev_layer)



    # Make a list of activations for each layer and return it.
    def feedforward(self, x):
        activations = []

        # NOTE: Ensure the number of nodes per layer are integers
        if type(self.layer_nums[0]) == int:
            assert x.shape[0] == self.layer_nums[0]
        else:
            assert x.shape == tuple(self.layer_nums[0])

        times = []
        for l in range(len(self.layers)):
            if l - 1 >= 0:
                prev_activation = activations[l-1]
            else:
                prev_activation = x

            activation = self.layers[l-1].call(prev_activation)
            activations.append(activation)

        return activation

    # NOTE: Evaluate the batch
    def evaluate(X, y):
            y_out = np.array([self.feedforward(x)[-1] for x in X])
            accuracy = np.mean(y_out.argmax(axis=1) == y.argmax(axis=1))
            loss = loss_function.call(y_out, y)


class Layer():
    def update_params(self, c):
        pass

    def update_nabla(self, prev_layer):
        self.nabla(prev_layer)

    def nabla(self, prev_layer):
        if self.param_num() == 0:
            return

    def update_params(self, c):
        pass


class Dense(Layer):
    def __init__(self, n, activation):
        self.n = n
        self. activation

    def build(self, input_num):
        assert type(input_num) != tuple

        self.weight = np.random.randn(self.n, input_num) / np.sqrt(input_num)
        self.bias = np.random.randn(self.n)

        self.weight_nabla = np.zeros_like(self.weight)
        self.bias_nabla = np.zeros_like(self.bias)

        self.input_num = input_num

    def call(self, x):
        weighted_input = np.dot(self.weight, x) + self.bias
        activation = self.activation_function.call(weighted_input)

        self.weighted_input = weighted_input
        self.activation = activation

        return activation

    def backprop(self, prev_layer):
        return np.dot(self.weight.T, self.error) * prev_layer.activation_function.prime(self.weighted_input)

    def nabla(self, prev_layer):
         w_nabla = np.array([prev_layer.activation] * self.n) * np.array([self.error] * self.input_num).T
         b_nabla = self.error

         return w_nabla, b_nabla

    def update_nabla(self, c):
        self.weight += c * self.weight_nabla
        self.bias += c * self.bias_nabla

        self.weight_nabla = 0
        self.bias_nabla = 0

    def output_num(self):
        return self.n

    def param_num(self):
        weight_num = self.weight.shape[0] * self.weight.shape[1]
        bias_num = self.bias.shape[0]

        return weight_num + bias_num


def Input(Layer):
    def __init (self, shape):
        self.shape = shape

    def build(self):
        pass

    def call(self, x):
        self.weighted_input = x
        self.activation = x
        return x

    def output_num(self):
        return self.shape
