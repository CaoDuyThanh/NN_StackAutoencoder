from __future__ import print_function
import timeit
import Image
import Utils.DataHelper as DataHelper
from pylab import *
from Layers.AELayer import *
from Utils.FilterHelper import *


# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
SAVE_PATH = '../Pretrained/model.pkl'
BATCH_SIZE = 1
VALIDATION_FREQUENCY = 10000
VISUALIZE_FREQUENCY = 500

# NETWORK
HIDDEN_LAYERS_SIZES = [1000, 1000, 1000]
CORRUPTIONS_LAYERS = [0.1, 0.2, 0.3]
NUM_OUT = 10

# Pretrain hyper parameters
PRETRAINING_EPOCH = 15
PRETRAINING_LEARNING_RATE = 0.001

# Fine-tuning hyper parameters
TRAINING_EPOCH = 1000
TRAINING_LEARNING_RATE = 0.1

''' Load dataset | Create Model | Evaluate Model '''
def StackAutoencoder():
    # Load datasets from local disk or download from the internet
    # We only load the images, not label
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX, trainSetY = datasets[0]
    validSetX, validSetY = datasets[1]
    testSetX, testSetY   = datasets[2]

    nTrainBatchs = trainSetX.get_value(borrow=True).shape[0]
    nValidBatchs = validSetX.get_value(borrow=True).shape[0]
    nTestBatchs = testSetX.get_value(borrow=True).shape[0]
    nTrainBatchs //= BATCH_SIZE
    nValidBatchs //= BATCH_SIZE
    nTestBatchs //= BATCH_SIZE

    ################################################
    #           Create encoder model and           #
    #           multiple perceptron model          #
    ################################################
    '''
    MODEL ARCHITECTURE

    INPUT    ->    HIDDEN LAYER    ->    HIDDEN LAYER    ->     HIDDEN LAYER    ->    OUTPUT
    ( 28x28 )    ( 1000 neurons )      ( 1000 neurons )       ( 1000 neurons )    ( 10 outputs )

    '''
    # Create random state
    rng = numpy.random.RandomState(123)

    # Create shared variable for input
    X = T.matrix('X')
    Y = T.ivector('Y')
    LearningRate = T.fscalar('LearningRate')

    X2D = X.reshape((BATCH_SIZE, 28 * 28))
    encoderLayers = []
    hiddenLayers = []
    for idx in range(len(HIDDEN_LAYERS_SIZES)):
        if (idx == 0):
            inputSize = 28 * 28
        else:
            inputSize = HIDDEN_LAYERS_SIZES[idx - 1]

        if (idx == 0):
            layerInput = X2D
        else:
            layerInput = hiddenLayers[-1].Output
        outputSize = HIDDEN_LAYERS_SIZES[idx]

        encoderLayer = AELayer(
            rng = rng,
            input = layerInput,
            numIn = inputSize,
            numOut = outputSize,
            learningRate = LearningRate,
            corruption = CORRUPTIONS_LAYERS[idx],
            activation = T.nnet.sigmoid
        )
        encoderLayerW = encoderLayer.Layers[0].W
        encoderLayerb = encoderLayer.Layers[0].b
        encoderLayers.append(encoderLayer)

        hiddenLayer = HiddenLayer(
            rng = rng,  # Random seed
            input = layerInput,  # Data input
            numIn = inputSize,  # Number neurons of input
            numOut = outputSize,  # Number reurons out of layer
            activation = T.nnet.sigmoid,  # Activation function
            W = encoderLayerW,
            b = encoderLayerb
        )
        hiddenLayers.append(hiddenLayer)
    # Add logistic layer on top of MLP
    hiddenLayer = HiddenLayer(
        rng = rng,  # Random seed
        input = hiddenLayers[-1].Output,  # Data input
        numIn = HIDDEN_LAYERS_SIZES[-1],  # Number neurons of input
        numOut = NUM_OUT,  # Number reurons out of layer
        activation = T.nnet.sigmoid  # Activation function
    )
    hiddenLayers.append(hiddenLayer)

    ################################################
    #           Calculate cost function            #
    ################################################
    # Encoder layer
    encoderLayersFunc = []
    for idx in range(len(HIDDEN_LAYERS_SIZES)):
        encoderLayer = encoderLayers[idx]

        # Train function
        trainFunc = theano.function(
            inputs = [X, LearningRate],
            outputs = [encoderLayer.CostFunc],
            updates = encoderLayer.Updates
        )

        # Valid function
        validFunc = theano.function(
            inputs=[X],
            outputs=[encoderLayer.CostFunc]
        )

        # Test function
        testFunc = theano.function(
            inputs = [X],
            outputs = [encoderLayer.CostFunc]
        )
        encoderLayersFunc.append([trainFunc, validFunc, testFunc])

    # Hidden layer
    hiddenLayersFunc = []
    hiddenLayer = hiddenLayers[-1]
    hiddenLayerOut = hiddenLayer.Output
    cost = CrossEntropy(hiddenLayerOut, Y)
    params = hiddenLayer.Params
    grads = T.grad(cost, params)
    updates = [(param, param - LearningRate * grad)
               for (param, grad) in zip(params, grads)]
    # Train function
    trainFunc = theano.function(
        inputs = [X, Y, LearningRate],
        outputs = [cost],
        updates = updates
    )

    # Train function
    testFunc = theano.function(
        inputs = [X, Y],
        outputs = [cost]
    )
    hiddenLayersFunc.append([trainFunc, testFunc])


    ################################################
    #           Training the model                 #
    ################################################
    iter = 0
    bestSumCost = 10000

    # Pre-training stage
    for epoch in range(PRETRAINING_EPOCH):
        for trainBatchIdx in range(nTrainBatchs):
            iter += 1
            subTrainSetX = trainSetX.get_value(borrow = True)[trainBatchIdx * BATCH_SIZE : (trainBatchIdx + 1) * BATCH_SIZE]
            costAELayer = [ encoderLayerFunc[0](subTrainSetX, PRETRAINING_LEARNING_RATE) for encoderLayerFunc in encoderLayersFunc]

            if iter % VISUALIZE_FREQUENCY == 0:
                print ('Epoch = %d, iteration = %d ' % (epoch, iter))
                for idx, costAE in enumerate(costAELayer):
                    print ('      CostAELayer %d = %f ' % (idx, costAE[0]))

            if iter % VALIDATION_FREQUENCY == 0:
                sumCost = 0
                print ('Validate current model ')
                for validBatchIdx in range(nValidBatchs):
                    subValidSetX = validSetX.get_value(borrow=True)[validBatchIdx * BATCH_SIZE: (validBatchIdx + 1) * BATCH_SIZE]
                    costAELayer = [encoderLayerFunc[1](subValidSetX) for encoderLayerFunc in encoderLayersFunc]
                    sumCost += sum(costAELayer)
                sumCost /= (len(encoderLayersFunc) * nValidBatchs)

                if (sumCost < bestSumCost):
                    print ('Save model !')
                    file = open(SAVE_PATH, 'wb')
                    [encoderLayer.SaveModel(file) for encoderLayer in encoderLayers]
                    file.close()










    # Fine-tuning stage

if __name__ == '__main__':
    StackAutoencoder()