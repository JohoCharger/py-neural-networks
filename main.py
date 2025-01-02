import math

import numpy as np
import util
import random

def main():
    t10k_image_path = "mnist/t10k-images.idx3-ubyte"  # Replace with your path
    t10k_label_path = "mnist/t10k-labels.idx1-ubyte"  # Replace with your path
    raw_data = util.load_idx3_ubyte(t10k_image_path)
    raw_labels = util.load_idx1_ubyte(t10k_label_path)

    data = raw_data.reshape((10000, 784))
    for i in range(len(data)):
        data[i] = data[i] / 255

    labels = np.zeros((10000, 10))
    for i in range(len(raw_labels)):
        label = raw_labels[i]
        one_hot = np.zeros(10)
        one_hot[label] = 1
        labels[i] = one_hot

    x = data[:9000]
    y = labels[:9000]

    w_1 = np.random.randn(784, 30)
    w_2 = np.random.randn(30, 10)
    b_1 = np.random.randn(30)
    b_2 = np.random.randn(10)

    learning_rate = 0.1
    epochs = 100
    batch_size = 10

    for epoch in range(epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Forward pass
            a_0 = x_batch
            z_1 = a_0 @ w_1 + b_1
            a_1 = sigmoid(z_1)
            z_2 = a_1 @ w_2 + b_2
            a_2 = sigmoid(z_2)

            # Compute cost
            c = cost(a_2, y_batch)

            # Backpropagation
            dC_dW2, dC_db2, dC_dZ2 = backprop_last_layer(batch_size, a_2, a_1, y_batch)
            dC_dW1, dC_db1, dC_dZ1 = backprop_middle_layer(dC_dZ2, w_2, a_0, z_1)

            # Update parameters
            w_1, w_2, b_1, b_2 = update_params(w_1, w_2, b_1, b_2, dC_dW1, dC_dW2, dC_db1, dC_db2, learning_rate)

        print(f"Epoch {epoch + 1} completed")
        test_x = data[9000:]
        test_y = labels[9000:]
        correct = 0
        for i in range(len(test_x)):
            a_0 = test_x[i]
            a_2 = feed_forward(a_0, w_1, w_2, b_1, b_2)
            if np.argmax(a_2) == np.argmax(test_y[i]):
                correct += 1

        print(f"Accuracy: {math.floor(correct / len(test_x) * 100)}%")



def feed_forward(a_0, w_1, w_2, b_1, b_2):
    z_1 = a_0 @ w_1 + b_1
    a_1 = sigmoid(z_1)
    z_2 = a_1 @ w_2 + b_2
    a_2 = sigmoid(z_2)
    return a_2

def sigmoid(m):
    return 1 / (1 + np.exp(-m))

def sigmoid_prime(m):
    return sigmoid(m) * (1 - sigmoid(m))

def cost(y_hat, y):
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )
    m = y_hat.reshape(-1).shape[0]
    summed_losses = (1 / m) * np.sum(losses, axis=1)
    return summed_losses

def backprop_last_layer(m, a_3, a_1, y):
    dC_dZ2 = (1/m) * (a_3 - y)
    dC_dW2 = a_1.T @ dC_dZ2
    dC_db2 = np.sum(dC_dZ2, axis=0)
    return dC_dW2, dC_db2, dC_dZ2

def backprop_middle_layer(dC_dZ2, w_2, a_0, z_1):
    dC_dA1 = dC_dZ2 @ w_2.T
    dC_dZ1 = dC_dA1 * sigmoid_prime(z_1)
    dC_dW1 = a_0.T @ dC_dZ1
    dC_db1 = np.sum(dC_dZ1, axis=0)
    return dC_dW1, dC_db1, dC_dZ1

def update_params(w_1, w_2, b_1, b_2, dC_dW1, dC_dW2, dC_db1, dC_db2, learning_rate):
    w_1 = w_1 - learning_rate * dC_dW1
    w_2 = w_2 - learning_rate * dC_dW2
    b_1 = b_1 - learning_rate * dC_db1
    b_2 = b_2 - learning_rate * dC_db2
    return w_1, w_2, b_1, b_2


if __name__ == "__main__":
    main()