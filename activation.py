from layer import Layer

class Activation(Layer) :
    def __init__(self, activation, activationPrime) :
        self.activation = activation
        self.activationPrime = activationPrime

    # returns activated output
    def fwdProp(self, input) :
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backProp(self, outputErr, lr) :
        return self.activationPrime(self.input) * outputErr
