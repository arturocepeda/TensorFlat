
###############################################################################
#
#  Copyright (c) 2023-2025 Arturo Cepeda Pérez
#
#  This software is provided 'as-is', without any express or implied
#  warranty. In no event will the authors be held liable for any damages
#  arising from the use of this software.
#
#  Permission is granted to anyone to use this software for any purpose,
#  including commercial applications, and to alter it and redistribute it
#  freely, subject to the following restrictions:
#
#  1. The origin of this software must not be misrepresented; you must not
#     claim that you wrote the original software. If you use this software
#     in a product, an acknowledgment in the product documentation would be
#     appreciated but is not required.
#
#  2. Altered source versions must be plainly marked as such, and must not be
#     misrepresented as being the original software.
#
#  3. This notice may not be removed or altered from any source distribution.
#
###############################################################################

import json
import keras
import numpy
import os.path
import pandas

from keras.models import Sequential
from keras.layers import Input, Dense

kKernelInitializer = "he_normal"

def loadNetworkDescriptionData(dataDirectory):
  jsonFilePath = os.path.join(dataDirectory, "nn.json")
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)
  
  descriptionData = jsonData["Description"]

  jsonFile.close()

  return descriptionData


def loadNetworkTrainingParameters(dataDirectory):
  jsonFilePath = os.path.join(dataDirectory, "nn.json")
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)
  
  trainingParameters = jsonData["Training"]

  jsonFile.close()

  return trainingParameters


def assertActivationDescription(activationFunctions, description):
  assert description in activationFunctions, "Unsupported activation function: '" + description + "'"


def createNetwork(dataDirectory):
  # Load data from the description file
  descriptionData = loadNetworkDescriptionData(dataDirectory)

  inputs = descriptionData["Inputs"]
  hiddenLayers = descriptionData["HiddenLayers"]
  outputs = descriptionData["Outputs"]

  activationFunctions = {}
  activationFunctions["Linear"] = "linear"
  activationFunctions["Sigmoid"] = "sigmoid"
  activationFunctions["ReLU"] = "relu"
  activationFunctions["LeakyReLU"] = keras.layers.LeakyReLU(negative_slope=descriptionData["LeakyReLUNegativeSlope"])

  outputLayerActivationDescription = descriptionData["OutputLayerActivation"]
  assertActivationDescription(activationFunctions, outputLayerActivationDescription)

  # Define the neural network models
  inputLayerSize = len(inputs)
  outputLayerSize = len(outputs)

  network = Sequential()

  inputLayer = Input(shape=(inputLayerSize,))
  network.add(inputLayer)

  for hiddenLayer in hiddenLayers:
    hiddenLayerSize = hiddenLayer["HiddenLayerSize"]
    hiddenLayerActivationDescription = hiddenLayer["HiddenLayerActivation"]
    assertActivationDescription(activationFunctions, hiddenLayerActivationDescription)

    hiddenLayer = Dense(
      hiddenLayerSize,
      activation=activationFunctions[hiddenLayerActivationDescription],
      kernel_initializer=kKernelInitializer
    )
    network.add(hiddenLayer)

  outputLayer = Dense(
    outputLayerSize,
    activation=activationFunctions[outputLayerActivationDescription],
    kernel_initializer=kKernelInitializer
  )
  network.add(outputLayer)
  
  # Load weights and biases, if available
  for layerIndex in range(len(network.layers)):
    weightsFilePath = os.path.join(dataDirectory, "_layer" + str(layerIndex) + "_weights_")
    biasesFilePath = os.path.join(dataDirectory, "_layer" + str(layerIndex) + "_biases_")

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

      network.layers[layerIndex].set_weights([weightsNP, biasNP])

  return network
