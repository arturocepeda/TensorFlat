
import imports.nn as nn

import sys
import pandas

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_predict.py <name>")
  exit(1)

name = sys.argv[1]

# Create the neural network
network = nn.createNetwork(name)
print(network.summary())

# Load the input data
inputLayerSize = network.input_shape[1]
dataSetX = pandas.read_csv(name + "/_inputs_", sep=" ", header=None)
inputs = dataSetX.iloc[:,:inputLayerSize].values

# Save the prediction
prediction = network.predict(inputs)

with open(name + "/_prediction_", "w") as file:
  for predictionSample in prediction:
    for value in predictionSample:
      file.write(str(value) + " ")
    
    file.write("\n")
