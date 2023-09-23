class Network :
    def __init__(self) :
        self.layers = []
        self.loss = None
        self.lossPrime = None

    def addLayer(self, layer) :
        self.layers.append(layer)

    def setLoss(self, loss, lossPrime) :
        self.loss = loss
        self.lossPrime = lossPrime

    def predict(self, input) :
        # sample dimension
        samples = len(input)
        result = []

        # run network over all samples
        for i in range(samples) :
            # forward propagation
            output = input[i]
            for layer in self.layers :
                output = layer.fwdProp(output)
            result.append(output)
        
        return result

    # train network
    def fit(self, xTrain, yTrain, epochs, lr) :
        # sample dimension
        samples = len(xTrain)

        # training loop
        for i in range(epochs) :
            err = 0
            for j in range(samples) :
                # forward propagation
                output = xTrain[j]
                for layer in self.layers :
                    output = layer.fwdProp(output)

                # compute loss (to display)
                err += self.loss(yTrain[j], output)

                # backward propagation
                error = self.lossPrime(yTrain[j], output)
                for layer in reversed(self.layers) :
                    error = layer.backProp(error, lr)

            # calculate average error on all samples
            err /= samples
            print('Epoch %d/%d\tError=%f' % (i+1, epochs, err))
