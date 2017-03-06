from __future__ import print_function
import timeit
import Image
import os
import Utils.DataHelper as DataHelper
from pylab import *
from Layers.AELayer import *
from Utils.FilterHelper import *


# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
SAVE_PATH = '../Pretrained/model_batch_50.pkl'
BATCH_SIZE = 1
VALIDATION_FREQUENCY = 10000
VISUALIZE_FREQUENCY = 500

# NETWORK
HIDDEN_LAYERS_SIZES = [1000, 1000, 1000]
CORRUPTIONS_LAYERS = [0.1, 0.2, 0.3]
NUM_OUT = 10

# Pretrain hyper parameters
PRETRAINING_SAVE_PATH = '../Pretrained/pretrain_stage.pkl'
PRETRAINING_EPOCH = 15
PRETRAINING_LEARNING_RATE = 0.001

# Fine-tuning hyper parameters
TRAINING_SAVE_PATH = '../Pretrained/training_stage.pkl'
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
    hiddenLayer = hiddenLayers[-1]
    hiddenLayerOut = hiddenLayer.Output
    cost = BinaryCrossEntropy(hiddenLayerOut, Y)
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
    fineTuningFunc = [trainFunc, testFunc]

    ################################################
    #           Training the model                 #
    ################################################
    # Load old model before training
    if os.path.isfile(PRETRAINING_SAVE_PATH):
        file = open(PRETRAINING_SAVE_PATH)
        [encoderLayer.LoadModel(file) for encoderLayer in encoderLayers]
        file.close()
    iter = 0

    # Pre-training stage
    for idx, encoderLayerFunc in enumerate(encoderLayersFunc):
        print ('Train layer %d ' % (idx))
        for epoch in range(PRETRAINING_EPOCH):
            for trainBatchIdx in range(nTrainBatchs):
                iter += BATCH_SIZE
                subTrainSetX = trainSetX.get_value(borrow = True)[trainBatchIdx * BATCH_SIZE : (trainBatchIdx + 1) * BATCH_SIZE]
                costAELayer = encoderLayerFunc[0](subTrainSetX, PRETRAINING_LEARNING_RATE)

                if iter % VISUALIZE_FREQUENCY == 0:
                    print ('Epoch = %d, iteration = %d ' % (epoch, iter))
                    print ('      CostAELayer = %f ' % (costAELayer[0]))

                if iter % VALIDATION_FREQUENCY == 0:
                    print ('Validate current model ')

                    validCost = 0
                    for validBatchIdx in range(nValidBatchs):
                        subValidSetX = validSetX.get_value(borrow=True)[validBatchIdx * BATCH_SIZE: (validBatchIdx + 1) * BATCH_SIZE]
                        validCost += encoderLayerFunc[1](subValidSetX)[0]
                    validCost /= nValidBatchs
                    print ('Validation cost = %f ' % (validCost))

                    file = open(SAVE_PATH, 'wb')
                    [encoderLayer.SaveModel(file) for encoderLayer in encoderLayers]
                    file.close()
                    print('Save model !')

    # Fine-tuning stage
    # Load save model
    if os.path.isfile(PRETRAINING_SAVE_PATH):
        file = open(PRETRAINING_SAVE_PATH)
        [encoderLayer.LoadModel(file) for encoderLayer in encoderLayers]
        file.close()
    shape = [(28, 28), (50, 20), (50, 20)]
    for idx, encoderLayer in enumerate(encoderLayers):
        image = Image.fromarray(tile_raster_images(
            X=encoderLayer.Params[0].get_value(borrow=True).T,
            img_shape=shape[idx], tile_shape=(10, 10),
            tile_spacing=(1, 1)))
        image.save('filters_corruption_30_%d.png' % (idx))
    # Traing.....
    iter = 0
    bestCost = 10000
    for epoch in range(TRAINING_EPOCH):
        for trainBatchIdx in range(nTrainBatchs):
            iter += BATCH_SIZE
            subTrainSetX = trainSetX.get_value(borrow = True)[trainBatchIdx * BATCH_SIZE: (trainBatchIdx + 1) * BATCH_SIZE]
            subTrainSetY = trainSetY.get_value(borrow = True)[trainBatchIdx * BATCH_SIZE: (trainBatchIdx + 1) * BATCH_SIZE]
            cost = fineTuningFunc[0](subTrainSetX, subTrainSetY, PRETRAINING_LEARNING_RATE)

            if iter % VISUALIZE_FREQUENCY == 0:
                print ('Epoch = %d, iteration = %d ' % (epoch, iter))
                print ('      Cost fine-tuning = %f ' % (cost))

            if iter % VALIDATION_FREQUENCY == 0:
                validCost = 0
                print ('Validate current model ')
                for validBatchIdx in range(nValidBatchs):
                    subValidSetX = validSetX.get_value(borrow=True)[validBatchIdx * BATCH_SIZE: (validBatchIdx + 1) * BATCH_SIZE]
                    subValidSetY = validSetY.get_value(borrow=True)[validBatchIdx * BATCH_SIZE: (validBatchIdx + 1) * BATCH_SIZE]
                    validCost += fineTuningFunc[1](subValidSetX, subValidSetY)
                validCost /= (nValidBatchs)

                if (validCost < bestCost):
                    bestCost = validCost
                    print ('Save model ! Sum cost = %f ' % (sumCost))
                    file = open(TRAINING_SAVE_PATH, 'wb')
                    [encoderLayer.SaveModel(file) for encoderLayer in encoderLayers]
                    hiddenLayers[-1].SaveModel(file)
                    file.close()

if __name__ == '__main__':
    StackAutoencoder()