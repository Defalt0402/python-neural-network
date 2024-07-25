import numpy as np

class Network:
    def __init__(self, inputNeurons, loss):
        self.hidden_layers = list()
        self.inputNeurons = inputNeurons
        self.loss = loss()
        self.alpha = 0.9

    def add_layer(self, numInputs, neurons, activation):
        self.hidden_layers.append(Layer(numInputs, neurons, activation))

    def forward(self, inputs):
        self.inputs = inputs
        currentOutputs = inputs

        for layer in self.hidden_layers:
            layer.forward(currentOutputs)
            currentOutputs = layer.outputs

        return currentOutputs

    def partial_fit(self, inputs, y):
        # Calculate predictions and loss for reporting and use
        self.current_predictions = self.forward(inputs)
        self.current_loss = self.loss.calculate(self.current_predictions, y)
        
        # Backward pass
        gradient = self.loss.backward(self.current_predictions, y)
        for layer in reversed(self.hidden_layers):
            gradient = layer.backward(gradient, self.alpha)

    def train(self, inputs, y, epochs, report=0):
        for i in range(epochs + 1):
            self.current_predictions = self.forward(inputs)
            self.current_loss = self.loss.calculate(self.current_predictions, y)
            
            # Backward pass
            gradient = self.loss.backward(self.current_predictions, y)
            for layer in reversed(self.hidden_layers):
                gradient = layer.backward(gradient, self.alpha)

            if report != 0:
                if i % report == 0:
                    print(f"Epoch {i}, Loss: {self.current_loss}")
                    print(f"Stats {self.get_stats(inputs, y)}")
                    print(f"{self.current_predictions}\n")

    def get_stats(self, inputs, y):
        self.current_predictions = self.forward(inputs)
        yPred = np.argmax(self.current_predictions, axis=1)

        stats = {}
        # tp + tn / all
        stats["accuracy"] = np.mean(yPred == y)
        # tp / tp + fp
        stats["precision"] = np.sum((yPred == y) & (yPred == 1)) / np.sum(yPred == 1)
        # tp / tp + fn
        stats["recall"] = np.sum((yPred == y) & (y == 1)) / np.sum(y == 1)
        # 2 * precision * recall / precision + recall
        stats["f1_score"] = 2 * (stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"])

        return stats


class Layer:
    def __init__(self, numInputs, neurons, activation):
        self.weights = np.random.rand(numInputs, neurons) * 0.01
        self.biases = np.zeros((1, neurons))
        self.activation = activation()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        self.activation.forward(self.outputs)
        self.outputs = self.activation.outputs

    def backward(self, gradient, learning_rate):
        self.activation.backward(gradient)
        gradient = self.activation.dinputs
        self.dweights = np.dot(self.inputs.T, gradient)
        self.dbiases = np.sum(gradient, axis=0, keepdims=True)
        
        self.dinputs = np.dot(gradient, self.weights.T)
        
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        
        return self.dinputs


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)

    def backward(self, gradient):
        self.dinputs = gradient.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        expVals = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.outputs = expVals / np.sum(expVals, axis=1, keepdims=True) 

    def backward(self, gradient):
        self.dinputs = gradient.copy() 

class Loss:
    def calculate(self, yPred, y):
        return np.mean(self.forward(yPred, y))
    
class Cross_Entropy_Loss(Loss):
    def forward(self, yPred, y):
        numSamples = len(yPred)
        yPredClip = np.clip(yPred, 1e-8, 1-1e-8)

        if len(y.shape) == 1:
            confidence = yPredClip[range(numSamples), y]
        elif len(y.shape) == 2:
            confidence = np.sum(yPredClip*y, axis=1)
        
        return -np.log(confidence)

    def backward(self, yPred, y):
        numSamples = len(yPred)
        if len(y.shape) == 1:
            labels = y
        elif len(y.shape) == 2:
            labels = np.argmax(y, axis=1)
        
        self.dinputs = yPred.copy()
        self.dinputs[range(numSamples), labels] -= 1
        self.dinputs = self.dinputs / numSamples
        return self.dinputs