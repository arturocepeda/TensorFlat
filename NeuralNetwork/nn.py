
import json
import keras
import numpy
import os.path
import pandas

from keras.models import Sequential
from keras.layers import Dense

kLeakyReLUAlpha = 0.01
kHiddenLayerActivation = keras.layers.LeakyReLU(alpha=kLeakyReLUAlpha)
kOutputLayerActivation = keras.layers.LeakyReLU(alpha=kLeakyReLUAlpha)

kKernelInitializer = "he_normal"

def createNetwork(name):
  # Load data from the description file
  jsonFilePath = name + "/nn.json"
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)
  
  inputs = jsonData["Inputs"]
  hiddenLayerSize = jsonData["HiddenLayerSize"]
  outputs = jsonData["Outputs"]

  jsonFile.close()

  # Define the neural network models
  inputLayerSize = len(inputs)
  outputLayerSize = len(outputs)

  network = Sequential()

  network.add(Dense(hiddenLayerSize, input_dim=inputLayerSize, activation=kHiddenLayerActivation, kernel_initializer=kKernelInitializer))
  network.add(Dense(outputLayerSize, activation=kHiddenLayerActivation, kernel_initializer=kKernelInitializer))
  
  # Load the weights, if available
  for layerIndex in range(len(network.layers)):
    weightsNP = numpy.empty(network.layers[layerIndex].get_weights()[0].shape)
    weightsFilePath = name + "/_layer" + str(layerIndex) + "_weights_"

    if os.path.exists(weightsFilePath):
      csvData = pandas.read_csv(weightsFilePath, sep=" ", header=None)
      layerWeightsLength = len(network.layers[layerIndex].get_weights()[0])
      layerWeights = csvData.iloc[:,:layerWeightsLength].values
      
      for x in range(layerWeightsLength):
        for y in range(len(network.layers[layerIndex].get_weights()[0][x])):
          weightsNP[x][y] = layerWeights[x][y]

    biasNP = numpy.empty(network.layers[layerIndex].get_weights()[1].shape)
    biasesFilePath = name + "/_layer" + str(layerIndex) + "_bias_"

    if os.path.exists(biasesFilePath):
      csvData = pandas.read_csv(biasesFilePath, sep=" ", header=None)
      layerBiasLength = len(network.layers[layerIndex].get_weights()[1])
      layerBias = csvData.iloc[:,:layerBiasLength].values
    
      for x in range(layerBiasLength):
        biasNP[x] = layerBias[0][x]
      
    network.layers[layerIndex].set_weights(numpy.array([weightsNP, biasNP], dtype=object))

  return network
