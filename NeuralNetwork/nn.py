
import keras

from keras.models import Sequential
from keras.layers import Dense

# TODO: load these values from a config file!
name = "NNTest"
inputs = [
  "InputValue001",
  "InputValue002",
  "InputValue003",
  "InputValue004",
  "InputValue005",
  "InputValue006",
  "InputValue007",
  "InputValue008"
]
hiddenLayerSize = 16
outputs = [
  "OutputValue001",
  "OutputValue002"
]

# Define the neural network models
inputLayerSize = len(inputs)
outputLayerSize = len(outputs)

kLeakyReLUAlpha = 0.01
kHiddenLayerActivation = keras.layers.LeakyReLU(alpha=kLeakyReLUAlpha)
kOutputLayerActivation = keras.layers.LeakyReLU(alpha=kLeakyReLUAlpha)

kKernelInitializer = "he_normal"

def createNetwork():
  network = Sequential()

  network.add(Dense(hiddenLayerSize, input_dim=inputLayerSize, activation=kHiddenLayerActivation, kernel_initializer=kKernelInitializer))
  network.add(Dense(outputLayerSize, activation=kHiddenLayerActivation, kernel_initializer=kKernelInitializer))
  
  return network
