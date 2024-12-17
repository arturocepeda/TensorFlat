
import imports.nn as nn

import sys
import os.path
import math
import pandas
import matplotlib.pyplot as pyplot

from keras.optimizers import Adam

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Script configuration
kPrintDataSet = False
kSaveWeights = True
kSavePrediction = False
kPlotTrainingHistoryAccuracy = True
kPlotTrainingHistoryLoss = True

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_train.py <data_directory>")
  exit(1)

dataDirectory = sys.argv[1]

# Define the neural network models
descriptionData = nn.loadNetworkDescriptionData(dataDirectory)

inputs = descriptionData["Inputs"]
outputs = descriptionData["Outputs"]

inputLayerSize = len(inputs)
outputLayerSize = len(outputs)

# Load the input and the output data
inputDataFilePath = os.path.join(dataDirectory, "_inputs_")
outputDataFilePath = os.path.join(dataDirectory, "_outputs_")

dataSetX = pandas.read_csv(inputDataFilePath, sep=" ", header=None)
inputData = dataSetX.iloc[:,:inputLayerSize].values
dataSetY = pandas.read_csv(outputDataFilePath, sep=" ", header=None)
outputData = dataSetY.iloc[:,:outputLayerSize].values

if kPrintDataSet:
  for i in range(len(inputData)):
    print("Sample #" + str(i))
    
    for j in range(inputLayerSize):
      print("  " + inputs[j] + ": " + str(inputData[i][j]))
      
    for j in range(outputLayerSize):
      print("  -> " + outputs[j] + ": " + str(outputData[i][j]))
      
    print("")
    
  input("Press Enter to continue...")

# Load the training parameters
trainingParameters = nn.loadNetworkTrainingParameters(dataDirectory)

testSetRatio = trainingParameters["TestSetRatio"]
trainingLearningRate = trainingParameters["LearningRate"]
trainingEpochs = trainingParameters["Epochs"]

# Shuffle the data
inputs, outputs = shuffle(inputData, outputData)

# Split the data
if math.isclose(testSetRatio, 0.0):
  testSetRatio = None

inputsTrain, inputsTest, outputsTrain, outputsTest = train_test_split(inputData, outputData, test_size=testSetRatio)

# Create the neural network
network = nn.createNetwork(dataDirectory)
print(network.summary())

opt = Adam(learning_rate=trainingLearningRate)
network.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

# Train the neural network
history = network.fit(inputsTrain, outputsTrain, validation_data=(inputsTest,outputsTest), epochs=trainingEpochs)

# Save the weights
if kSaveWeights:
  for layerIndex in range(len(network.layers)):
    weightsArray = network.layers[layerIndex].get_weights()[0]
    
    with open(os.path.join(dataDirectory, "_layer" + str(layerIndex) + "_weights_"), "w") as file:
      for i in range(len(weightsArray)):
        for j in range(len(weightsArray[i])):
          file.write(str(weightsArray[i][j]) + " ")
          
        file.write("\n")
        
    bias = network.layers[layerIndex].get_weights()[1]
    
    with open(os.path.join(dataDirectory, "_layer" + str(layerIndex) + "_biases_"), "w") as file:
      for i in range(len(bias)):
        file.write(str(bias[i]) + " ")
  
# Save the prediction
if kSavePrediction:
  prediction = network.predict(inputData)
  
  with open(os.path.join(dataDirectory, "_prediction_after_training_"), "w") as file:
    for predictionSample in prediction:
      for value in predictionSample:
        file.write(str(value) + " ")
      
      file.write("\n")

# Plot the training history data
if kPlotTrainingHistoryAccuracy:
  pyplot.plot(history.history["accuracy"])
  pyplot.plot(history.history["val_accuracy"])
  pyplot.title("Model accuracy")
  pyplot.ylabel("Accuracy")
  pyplot.xlabel("Epoch")
  pyplot.legend(["Train", "Test"], loc="upper left")
  pyplot.show()

if kPlotTrainingHistoryLoss:
  pyplot.plot(history.history["loss"])
  pyplot.plot(history.history["val_loss"]) 
  pyplot.title("Model loss") 
  pyplot.ylabel("Loss") 
  pyplot.xlabel("Epoch") 
  pyplot.legend(["Train", "Test"], loc="upper left") 
  pyplot.show()
