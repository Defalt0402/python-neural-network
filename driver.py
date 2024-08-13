from neural_net import *
import numpy as np
import struct

## Loads mnist images into an array of size(number of images, 28, 28)
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

## Loads mnist labels 
def read_labels(filePath):
    with open(filePath, 'rb') as f:
        # Read the header
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]
        
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
        
    return labels

## Encodes labels into one hot format
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

## Load images and labels for testing
testImages = read_mnist("MNIST_ORG/t10k-images.idx3-ubyte")
testLabels = read_labels("MNIST_ORG/t10k-labels.idx1-ubyte")
testImages = testImages.reshape(testImages.shape[0], -1).astype(np.float32) / 255.0

## Loads and tests the network
net = Network(0, Cross_Entropy_Loss)
net = net.load_model("MNIST_32x10")

stats = net.get_stats(testImages, testLabels)
print(stats)