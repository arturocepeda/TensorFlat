
import nn
import sys
import pandas
import matplotlib.pyplot as pyplot

from keras.optimizer_v2.adam import Adam
from keras.optimizer_experimental.sgd import SGD

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Script configuration
kPrintDataSet = False
kSaveWeights = True
kSavePrediction = False
kPlotTrainingHistoryAccuracy = True
kPlotTrainingHistoryLoss = True

# Training configuration
kTestSetRatio = 0.3

# Argument check
if len(sys.argv) < 4:
  print("Usage: nn_train.py <name> <trainingLearningRate> <trainingEpochs>")
  exit(1)

name = sys.argv[1]

# Define the neural network models
inputs, hiddenLayerSize, outputs = nn.loadNetworkDescription(name)
inputLayerSize = len(inputs)
outputLayerSize = len(outputs)

# Load the input and the output data
inputDataFilePath = name + "/_inputs_"
outputDataFilePath = name + "/_outputs_"

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

# Shuffle the data
inputs, outputs = shuffle(inputData, outputData)

# Split the data
inputsTrain, inputsTest, outputsTrain, outputsTest = train_test_split(inputData, outputData, test_size=kTestSetRatio)

# Create the neural network
network = nn.createNetwork(name)
print(network.summary())

trainingLearningRate = float(sys.argv[2])
opt = Adam(learning_rate=trainingLearningRate)
network.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

# Train the neural network
trainingEpochs = int(sys.argv[3])
history = network.fit(inputsTrain, outputsTrain, validation_data=(inputsTest,outputsTest), epochs=trainingEpochs)

# Save the weights
if kSaveWeights:
  for layerIndex in range(len(network.layers)):
    weightsArray = network.layers[layerIndex].get_weights()[0]
    
    with open(name + "/_layer" + str(layerIndex) + "_weights_", "w") as file:
      for i in range(len(weightsArray)):
        for j in range(len(weightsArray[i])):
          file.write(str(weightsArray[i][j]) + " ")
          
        file.write("\n")
        
    bias = network.layers[layerIndex].get_weights()[1]
    
    with open(name + "/_layer" + str(layerIndex) + "_bias_", "w") as file:
      for i in range(len(bias)):
        file.write(str(bias[i]) + " ")
  
# Save the prediction
if kSavePrediction:
  prediction = network.predict(inputs)
  
  with open(name + "/_prediction_after_training_", "w") as file:
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
