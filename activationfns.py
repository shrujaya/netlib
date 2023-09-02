import numpy as np

# activation function and its derivative

def tanh(x) :
    return np.tanh(x)

def tanhPrime(x) :
    return 1 - np.tanh(x)**2
