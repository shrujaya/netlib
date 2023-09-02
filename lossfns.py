import numpy as np

# loss function
def mse(yTrue, yPred) : 
    return np.mean(np.power(yTrue-yPred, 2))

# derivative
def msePrime(yTrue, yPred) :
    return 2 * (yPred-yTrue)/yTrue.size
