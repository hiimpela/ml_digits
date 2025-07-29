import numpy as np
import random as rand

class network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
        self.weights = [np.random.randn(r, c) for c, r in zip(sizes[:-1], sizes[1:])]

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def feedforward(self, a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a) + bias)
        return a
    
    def backpropagate(self, x, y):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        activation = x
        activations = [x]
        z_vectors = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_vectors.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        #backward pass
        error = self.cost_derivative(activations[-1], y) * sigmoid_prime(z_vectors[-1])
        nabla_b[-1] = error
        nabla_w[-1] = np.dot(error, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = z_vectors[-l]
            error = np.dot(self.weights[-l + 1].transpose(), error) * sigmoid_prime(z)

            nabla_b[-l] = error
            nabla_w[-l] = np.dot(error, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagate(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/ len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        n = len(training_data)
        if test_data: n_test = len(test_data)

        for epoch in range(epochs):
            rand.shuffle(training_data)
            mini_batches = [
                training_data[x:x + mini_batch_size] for x in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("epoch {0}: {1} / {2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("epoch {0} complete".format(epoch))
    
def predict(net, input):
    input_vector = np.reshape(input, (784, 1))
    output = net.feedforward(input)
    return output


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))