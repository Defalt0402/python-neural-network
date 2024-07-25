from neural_net import *
import numpy as np
import struct

def read_mnist(filePath):
    with open(filePath, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        
    return images

def read_labels(filePath):
    with open(filePath, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
        
    return labels

images = read_mnist("MNIST_ORG/t10k-images.idx3-ubyte")
labels = read_labels("MNIST_ORG/t10k-labels.idx1-ubyte")

images = images.reshape(images.shape[0], -1)

print(images.shape)  # Should be (10000, 784)
print(labels.shape)

# net = Network(2, Cross_Entropy_Loss)
# net.add_layer(2, 3, ReLU)
# net.add_layer(3, 3, ReLU)
# net.add_layer(3, 2, Softmax)

# inputs = np.array([[0, 1], [0, 0], [1, 2], [2, 2], [2, 1], [2, 3]])
# labels = np.array([0, 0, 1, 1, 1, 1])

# net.train(inputs, labels, 1000, 3)
# net.plot_metrics("test")

# net.save_model("test")

# net2 = Network(2, Cross_Entropy_Loss).load_model("test")

# predictions = net2.forward(inputs)
# print(net2.get_stats(inputs, labels))
# print(predictions)