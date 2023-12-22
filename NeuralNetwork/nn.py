
import json
import keras

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
  
  return network
