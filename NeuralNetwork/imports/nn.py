
import json
import keras
import numpy
import os.path
import pandas

from keras.models import Sequential
from keras.layers import Dense

kKernelInitializer = "he_normal"

def loadNetworkDescriptionData(name):
  jsonFilePath = name + "/nn.json"
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)
  
  descriptionData = jsonData["Description"]

  jsonFile.close()

  return descriptionData


def loadNetworkTrainingParameters(name):
  jsonFilePath = name + "/nn.json"
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)
  
  trainingParameters = jsonData["Training"]

  jsonFile.close()

  return trainingParameters


def assertActivationDescription(description):
  assert \
    description == "ReLU" or \
    description == "LeakyReLU" or \
    description == "Sigmoid", \
    "Unsupported activation function: '" + description + "'"


def createNetwork(name):
  # Load data from the description file
  descriptionData = loadNetworkDescriptionData(name)

  inputs = descriptionData["Inputs"]
  hiddenLayerSize = descriptionData["HiddenLayerSize"]
  outputs = descriptionData["Outputs"]

  hiddenLayerActivationDescription = descriptionData["HiddenLayerActivation"]
  assertActivationDescription(hiddenLayerActivationDescription)
  outputLayerActivationDescription = descriptionData["OutputLayerActivation"]
  assertActivationDescription(outputLayerActivationDescription)

  leakyReLUAlpha = descriptionData["LeakyReLUAlpha"]
  leakyReLU = keras.layers.LeakyReLU(alpha=leakyReLUAlpha)

  # Define the neural network models
  inputLayerSize = len(inputs)
  outputLayerSize = len(outputs)

  network = Sequential()

  if hiddenLayerActivationDescription == "ReLU":
    hiddenLayer = Dense(
      hiddenLayerSize, input_dim=inputLayerSize, activation="relu", kernel_initializer=kKernelInitializer)
  elif hiddenLayerActivationDescription == "LeakyReLU":
    hiddenLayer = Dense(
      hiddenLayerSize, input_dim=inputLayerSize, activation=leakyReLU, kernel_initializer=kKernelInitializer)
  else:
    hiddenLayer = Dense(
      hiddenLayerSize, input_dim=inputLayerSize, activation="sigmoid", kernel_initializer=kKernelInitializer)

  network.add(hiddenLayer)

  if outputLayerActivationDescription == "ReLU":
    outputLayer = Dense(outputLayerSize, activation="relu", kernel_initializer=kKernelInitializer)
  elif outputLayerActivationDescription == "LeakyReLU":
    outputLayer = Dense(outputLayerSize, activation=leakyReLU, kernel_initializer=kKernelInitializer)
  else:
    outputLayer = Dense(outputLayerSize, activation="sigmoid", kernel_initializer=kKernelInitializer)

  network.add(outputLayer)
  
  # Load weights and biases, if available
  for layerIndex in range(len(network.layers)):
    weightsFilePath = name + "/_layer" + str(layerIndex) + "_weights_"
    biasesFilePath = name + "/_layer" + str(layerIndex) + "_biases_"

    if os.path.exists(weightsFilePath) and os.path.exists(biasesFilePath):
      weightsNP = numpy.empty(network.layers[layerIndex].weights[0].shape)
      layerWeights = pandas.read_csv(weightsFilePath, sep=" ", header=None).to_numpy()

      for x in range(weightsNP.shape[0]):
        for y in range(weightsNP.shape[1]):
          weightsNP[x][y] = layerWeights[x][y]

      biasNP = numpy.empty(network.layers[layerIndex].weights[1].shape)
      layerBiases = pandas.read_csv(biasesFilePath, sep=" ", header=None).to_numpy()
    
      for x in range(biasNP.shape[0]):
        biasNP[x] = layerBiases[0][x]

      network.layers[layerIndex].set_weights(numpy.array([weightsNP, biasNP], dtype=object))

  return network
