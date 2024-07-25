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

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

trainImages = read_mnist("MNIST_ORG/train-images.idx3-ubyte")
trainLabels = read_labels("MNIST_ORG/train-labels.idx1-ubyte")
trainImages = trainImages.reshape(trainImages.shape[0], -1).astype(np.float32) / 255.0
# trainLabels = one_hot_encode(trainLabels, 10)

testImages = read_mnist("MNIST_ORG/t10k-images.idx3-ubyte")
testLabels = read_labels("MNIST_ORG/t10k-labels.idx1-ubyte")
testImages = testImages.reshape(testImages.shape[0], -1).astype(np.float32) / 255.0
# testLabels = one_hot_encode(testLabels, 10)

print(trainLabels)

net = Network(784, Cross_Entropy_Loss)
net.add_layer(784, 32, ReLU)
net.add_layer(32, 10, Softmax)

net.train(trainImages[0:10000], trainLabels[0:10000], 1000, 3)
net.plot_metrics("MNIST first run, (32, 32, 16, 10), 1000 epoch")

# inputs = np.array([[0, 1], [0, 0], [1, 2], [2, 2], [2, 1], [2, 3]])
# labels = np.array([0, 0, 1, 1, 1, 1])

# net.train(inputs, labels, 1000, 3)
# net.plot_metrics("test")

# net.save_model("test")

# net2 = Network(2, Cross_Entropy_Loss).load_model("test")

# predictions = net2.forward(inputs)
# print(net2.get_stats(inputs, labels))
# print(predictions)