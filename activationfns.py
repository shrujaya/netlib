import numpy as np

# activation function
def tanh(x) : 
    return np.tanh(x)

# derivative
def tanhPrime(x) :
    return 1 - np.tanh(x)**2
