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
