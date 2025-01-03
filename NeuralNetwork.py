import numpy as np
import activation_functions as af

class NeuralNetwork:
    def __init__(self, shape, activations, alpha):
        self.weights = []
        self.biases = []

        self._init_weights_and_biases(shape)
        self.layers = len(shape) - 1

        self.cache_a = [x for x in range(self.layers)]
        self.cache_z = [x for x in range(self.layers)]

        self.alpha = alpha

        self.a = [af.get_activation(act) for act in activations]
        self.a_primes = [af.get_activation_prime(act) for act in activations]

    
    def _init_weights_and_biases(self, shape):
        for i in range(len(shape) - 1):
            self.weights.append(np.random.randn(shape[i], shape[i+1]) * np.sqrt(2/(shape[i])))
            self.biases.append(np.zeros((1, shape[i+1])))

    def forward(self, x):
        for i in range(self.layers):
            z = x @ self.weights[i] + self.biases[i]
            a = self.a[i](z)
            self.cache_a[i] = a
            self.cache_z[i] = z
            x = a
        return x


    def backprop(self, x, y):
        m = x.shape[0]
        dz = (self.cache_a[self.layers - 1] - y) / m

        for i in range(self.layers - 1, -1, -1):
            if i > 0:
                dw = self.cache_a[i-1].T @ dz
            else:
                dw = x.T @ dz
            
            self.weights[i] -= self.alpha * dw
            db = np.sum(dz, axis=0, keepdims=True)
            self.biases[i] -= self.alpha * db
            if i > 0:
                dz = dz @ self.weights[i].T * self.a_primes[i - 1](self.cache_z[i - 1])  # Apply derivative to pre-activation values

    def test(self, x_test, y_test):
        correct = 0.0
        for i in range(len(x_test)):
            output = self.forward(x_test[i])

            if np.argmax(output) == np.argmax(y_test[i]):
                correct += 1.0

        return correct / float(len(x_test))
    
    def dump(self):
        for i, weight in enumerate(self.weights):
            np.savetxt(f"w_{i+1}.txt", weight)
        for i, bias in enumerate(self.biases):
            np.savetxt(f"b_{i+1}.txt", bias)
