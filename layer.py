class Layer : 
    def __init__(self) :
        self.input = None;
        self.output = None;

    def fwdProp(self, input) :
        raise NotImplementedError

    def backProp(self, outputErr, lr) :
        raise NotImplementedError
