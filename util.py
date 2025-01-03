import numpy as np
import struct

def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Invalid magic number: {magic}"

        # Read image data
        data = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return data


def load_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        # Read header information
        magic, num_data = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Invalid magic number: {magic}"

        # Read image data
        data = np.fromfile(f, dtype=np.uint8)
    return data

def prep_data():
    t10k_image_path = "mnist/t10k-images.idx3-ubyte"  # Replace with your path
    t10k_label_path = "mnist/t10k-labels.idx1-ubyte"  # Replace with your path
    raw_data = load_idx3_ubyte(t10k_image_path)
    raw_labels = load_idx1_ubyte(t10k_label_path)

    data = raw_data.reshape((10000, 784))
    data = data.astype(np.float32) / 255.0

    labels = np.zeros((10000, 10))
    for i in range(len(raw_labels)):
        label = raw_labels[i]
        one_hot = np.zeros(10)
        one_hot[label] = 1
        labels[i] = one_hot
        
    return data, labels
