import numpy as np

from network import Network
from dense import Dense
from activation import Activation
from activationfns import tanh, tanhPrime
from lossfns import mse, msePrime

# training data
xTrain = np.array([ [[0,0]], [[0,1]], [[1,0]], [[1,1]] ])
yTrain = np.array([ [[0]], [[1]], [[1]], [[0]] ])

# network
net = Network()
net.addLayer(Dense(2, 3))
net.addLayer(Activation(tanh, tanhPrime))
net.addLayer(Dense(3, 1))
net.addLayer(Activation(tanh, tanhPrime))

# train
net.setLoss(mse, msePrime)
net.fit(xTrain, yTrain, epochs=1000, lr=0.1)

# test
out = net.predict(xTrain)
print(out)
