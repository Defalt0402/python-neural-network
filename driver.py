from neural_net import *
import numpy as np

net = Network(2, Cross_Entropy_Loss)
net.add_layer(2, 3, ReLU)
net.add_layer(3, 3, ReLU)
net.add_layer(3, 2, Softmax)

inputs = np.array([[0, 1], [0, 0], [1, 2], [2, 2], [2, 1], [2, 3]])
labels = np.array([0, 0, 1, 1, 1, 1])

epochs = 1000
for epoch in range(epochs):
    # Forward pass
    predictions = net.forward(inputs)
    
    # Calculate loss
    loss = net.loss.calculate(predictions, labels)
    
    # Backward pass
    gradient = net.loss.backward(predictions, labels)
    net.backward(gradient)
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")
        print(predictions)

predictions = net.forward(inputs)
loss = net.loss.calculate(predictions, labels)
print(f"Final Loss: {loss}")