
import nn
import sys
import pandas
import numpy

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_predict.py <dataSetInputsFile>")
  exit(1)

# Load the input data
dataSetX = pandas.read_csv(sys.argv[1], sep=" ", header=None)
inputs = dataSetX.iloc[:,:nn.kInputLayerSize].values

# Create the neural network
network = nn.createNetwork()
print(network.summary())

# Load the weights
for layerIndex in range(len(network.layers)):
  weightsNP = numpy.empty(network.layers[layerIndex].get_weights()[0].shape)
  csvData = pandas.read_csv("./layer" + str(layerIndex) + "_weights", sep=" ", header=None)
  layerWeightsLength = len(network.layers[layerIndex].get_weights()[0])
  layerWeights = csvData.iloc[:,:layerWeightsLength].values
  
  for x in range(layerWeightsLength):
    for y in range(len(network.layers[layerIndex].get_weights()[0][x])):
      weightsNP[x][y] = layerWeights[x][y]

  biasNP = numpy.empty(network.layers[layerIndex].get_weights()[1].shape)  
  csvData = pandas.read_csv("./layer" + str(layerIndex) + "_bias", sep=" ", header=None)
  layerBiasLength = len(network.layers[layerIndex].get_weights()[1])
  layerBias = csvData.iloc[:,:layerBiasLength].values
  
  for x in range(layerBiasLength):
    biasNP[x] = layerBias[0][x]
    
  network.layers[layerIndex].set_weights(numpy.array([weightsNP, biasNP], dtype=object))
  
# Save the prediction
prediction = network.predict(inputs)

with open("./prediction", "w") as file:
  for predictionSample in prediction:
    for value in predictionSample:
      file.write(str(value) + " ")
    
    file.write("\n")
