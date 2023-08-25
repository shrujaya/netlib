from layer import Layer
import numpy as np

# derived from Layer
class Dense(Layer) :
    def __init__(self, inputSize, outputSize) :
        self.weights = np.random.rand(inputSize, outputSize) - 0.5
        self.bias = np.random.rand(1, outputSize) - 0.5

    def fwdProp(self, input) :
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backProp(self, err, lr) :
