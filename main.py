import numpy as np
from NeuralNetwork import NeuralNetwork
import util

def main():
    data, labels = util.prep_data()

    x = data[:9000]
    y = labels[:9000]

    nn = NeuralNetwork(
        input_size=784, 
        hidden_size=30, 
        output_size=10, 
        activations=["relu", "softmax"], 
        alpha=0.04
    )
    
    epochs = 2000
    batch_size = 30

    for epoch in range(epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            nn.forward(x_batch)
            nn.backprop(x_batch, y_batch)
                
        test_x = data[9900:]
        test_y = labels[9900:]
        
        accuracy = nn.test(test_x, test_y)

        print(f"Accuracy: {accuracy * 100}%")


def cost(y_hat, y):
    losses = - ( (y * np.log(y_hat)) + (1 - y)*np.log(1 - y_hat) )
    m = y_hat.reshape(-1).shape[0]
    summed_losses = (1 / m) * np.sum(losses, axis=1)
    return summed_losses

def backprop_last_layer(m, a_2, a_1, y):
    #dC_dZ2 = (1/m) * (a_2 - y)
    dC_dZ2 = (a_2 - y) / m
    dC_dW2 = a_1.T @ dC_dZ2
    dC_db2 = np.sum(dC_dZ2, axis=0)
    return dC_dW2, dC_db2, dC_dZ2

def backprop_middle_layer(dC_dZ2, w_2, a_0, z_1):
    dC_dA1 = dC_dZ2 @ w_2.T
    dC_dZ1 = dC_dA1 * relu_prime(z_1)
    dC_dW1 = a_0.T @ dC_dZ1
    dC_db1 = np.sum(dC_dZ1, axis=0)
    return dC_dW1, dC_db1, dC_dZ1


if __name__ == "__main__":
    main()