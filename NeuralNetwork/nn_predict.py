
import imports.nn as nn

import sys
import os.path
import pandas

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_predict.py <data_directory>")
  exit(1)

dataDirectory = sys.argv[1]

# Create the neural network
network = nn.createNetwork(dataDirectory)
print(network.summary())

# Load the input data
inputLayerSize = network.input_shape[1]
dataSetX = pandas.read_csv(os.path.join(dataDirectory, "_inputs_"), sep=" ", header=None)
inputs = dataSetX.iloc[:,:inputLayerSize].values

# Save the prediction
prediction = network.predict(inputs)

with open(os.path.join(dataDirectory, "_prediction_"), "w") as file:
  for predictionSample in prediction:
    for value in predictionSample:
      file.write(str(value) + " ")
    
    file.write("\n")
