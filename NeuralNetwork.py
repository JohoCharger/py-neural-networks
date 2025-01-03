import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activations, alpha):
        self.w_1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.w_2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b_1 = np.zeros((1, hidden_size))
        self.b_2 = np.zeros((1, output_size))
        self.cache = {
            "Z1": None,
            "Z2": None,
            "A1": None,
            "A2": None,
        }
        self.alpha = alpha

        match activations[0]:
            case "sigmoid":
                self.activation1 = _sigmoid
                self.activation1_prime = _sigmoid_prime
            case "relu":
                self.activation1 = _relu
                self.activation1_prime = _relu_prime
            case "softmax":
                self.activation1 = _softmax
                self.activation1_prime = None
            case _:
                raise ValueError("Invalid activation function")

        match activations[1]:
            case "sigmoid":
                self.activation2 = _sigmoid
                self.activation2_prime = _sigmoid_prime
            case "relu":
                self.activation2 = _relu
                self.activation2_prime = _relu_prime
            case "softmax":
                self.activation2 = _softmax
                self.activation2_prime = None
            case _:
                raise ValueError("Invalid activation function")

    def forward(self, x):
        z_1 = x @ self.w_1 + self.b_1
        self.cache["Z1"] = z_1
        a_1 = self.activation1(z_1)
        self.cache["A1"] = a_1
        z_2 = a_1 @ self.w_2 + self.b_2
        self.cache["Z2"] = z_2
        a_2 = self.activation2(z_2)
        self.cache["A2"] = a_2

    def backprop(self, x, y):
        m = x.shape[0]
        dz_2 = self.cache["A2"] - y
        dw_2 = (self.cache["A1"].T @ dz_2) / m
        db_2 = np.sum(dz_2, axis=0, keepdims=True) / m
        dz_1 = dz_2 @ self.w_2.T * self.activation1_prime(self.cache["Z1"])
        dw_1 = x.T @ dz_1 / m
        db_1 = np.sum(dz_1, axis=0, keepdims=True) / m

        self.w_1 -= self.alpha * dw_1
        self.w_2 -= self.alpha * dw_2
        self.b_1 -= self.alpha * db_1
        self.b_2 -= self.alpha * db_2


    def test(self, x_test, y_test):
        correct = 0
        for i in range(len(x_test)):
            self.forward(x_test[i])
            if np.argmax(self.cache["A2"]) == np.argmax(y_test[i]):
                correct += 1

        return correct / len(x_test)
    
    def dump(self):
        np.savetxt("w_1.txt", self.w_1)

def _sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def _sigmoid_prime(Z):
    return _sigmoid(Z) * (1 - _sigmoid(Z))

def _relu(Z):
    return np.maximum(0, Z)

def _relu_prime(Z):
    return Z > 0

def _softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
