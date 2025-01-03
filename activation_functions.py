import numpy as np

def _sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def _sigmoid_prime(Z):
    return _sigmoid(Z) * (1 - _sigmoid(Z))

def _relu(Z):
    return np.maximum(0, Z + 1e-7)

def _relu_prime(Z):
    return Z > 0

def _softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def _no_op(Z):
    return Z

def get_activation(activation):
    if activation == "sigmoid":
        return _sigmoid
    elif activation == "relu":
        return _relu
    elif activation == "softmax":
        return _softmax

def get_activation_prime(activation):
    if activation == "sigmoid":
        return _sigmoid_prime
    elif activation == "relu":
        return _relu_prime
    elif activation == "softmax":
        return _no_op