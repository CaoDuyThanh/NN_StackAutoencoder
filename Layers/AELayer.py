from Layers.HiddenLayer import *
from Utils.CostFHelper import *

class AELayer:
    def __init__(self,
                rng,
                input,
                numIn,
                numOut,
                learningRate,
                corruption = 0.0,
                activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.Input = input
        self.NumIn = numIn
        self.NumOut = numOut
        self.LearningRate = learningRate
        self.Corruption = corruption
        self.Activation = activation
        self.Layers = []

        self.createModel()

    def createModel(self):
        # Hidden layer 0
        hidLayer0 = HiddenLayer(
            rng = self.Rng,
            input = self.Input,
            numIn = self.NumIn,
            numOut = self.NumOut,
            corruption =  self.Corruption,
            activation = self.Activation
        )
        hidLayer0Output = hidLayer0.Output
        hidLayer0Params = hidLayer0.Params
        hidLayer0WTranspose = hidLayer0.WTranspose
        self.Layers.append(hidLayer0)

        # Hidden layer 1
        hidLayer1 = HiddenLayer(
            rng = self.Rng,
            input = hidLayer0Output,
            numIn = self.NumOut,
            numOut = self.NumIn,
            activation = self.Activation,
            W = hidLayer0WTranspose
        )
        hidLayer1Output = hidLayer1.Output
        hidLayer1Params = [hidLayer1.b]
        self.Layers.append(hidLayer1)

        # Parameters of model
        self.Params = hidLayer0Params + \
                      hidLayer1Params

        # Output of model
        self.Output = hidLayer1Output

        # Cost function
        self.CostFunc = BinaryCrossEntropy(self.Output, self.Input)

        # Update function
        grads = T.grad(self.CostFunc, self.Params)
        self.Updates = [(param, param - self.LearningRate * grad)
                        for (param, grad) in zip(self.Params, grads)]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self.Params]

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow=True) for param in self.Params]
