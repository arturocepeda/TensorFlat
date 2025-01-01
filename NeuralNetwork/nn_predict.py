
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
