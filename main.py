import numpy as np
from NeuralNetwork import NeuralNetwork
import util

def main():
    data, labels = util.prep_data()

    data = data / np.max(data)  # Normalize input data

    x = data[:9000]
    y = labels[:9000]

    nn = NeuralNetwork(
        shape=(784, 40, 40, 40, 40, 10),
        activations=["relu", "sigmoid", "relu", "sigmoid", "softmax"], 
        alpha=0.04
    )
    
    epochs = 100
    batch_size = 30

    for epoch in range(epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            nn.forward(x_batch)
            nn.backprop(x_batch, y_batch)
                
        test_x = data[9000:]
        test_y = labels[9000:]
        
        accuracy = nn.test(test_x, test_y)

        print("Epoch: {}, Accuracy = {:.2f}%".format(epoch + 1, accuracy * 100))

if __name__ == "__main__":
    main()