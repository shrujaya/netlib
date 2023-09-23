import numpy as np

from network import Network
from dense import Dense
from activation import Activation
from activationfns import tanh, tanhPrime
from lossfns import mse, msePrime

from keras.datasets import mnist
from keras.utils import to_categorical

# load MNIST from server
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

# reshape and normalize train data = 60000 samples
xTrain = xTrain.reshape(xTrain.shape[0], 1, 28*28)
xTrain = xTrain.astype('float32')
xTrain /= 255

# encode output in range [0,9] into a vector of size 10
# eg: 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
yTrain = to_categorical(yTrain)

# same for train data = 10000 samples
xTest = xTest.reshape(xTest.shape[0], 1, 28*28)
xTest = xTest.astype('float32')
xTest /= 255
yTest = to_categorical(yTest)

# build network
net = Network()
net.addLayer(Dense(28*28, 100)) # inputShape=(1,28*28) , outputShape=(1,100)
net.addLayer(Activation(tanh, tanhPrime))
net.addLayer(Dense(100, 50)) # inputShape=(1,100) , outputShape=(1,50)
net.addLayer(Activation(tanh, tanhPrime))
net.addLayer(Dense(50, 10)) # inputShape=(1,50) , outputShape=(1,10)
net.addLayer(Activation(tanh, tanhPrime))

# train on 1000 samples, slow if we update on each iteration for 60000 samples
net.setLoss(mse, msePrime)
net.fit(xTrain[0:1000], yTrain[0:1000], epochs=35, lr=0.1)

# test on 3 samples
out = net.predict(xTest[0:3])
print('\n')
print('Predicted Values: ')
print(out, end='\n')
print('True Values: ')
print(yTest[0:3])
