
import nn
import sys
import pandas
import keras
import numpy
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

# Argument check
if len(sys.argv) < 5:
  print("Usage: nn_train.py <dataSetInputsFile> <dataSetOutputsFile> <trainingLearningRate> <trainingEpochs>")
  exit(1)

kTestSetRatio = 0.3

# Load the input and the output data
dataSetX = pandas.read_csv(sys.argv[1], sep=" ", header=None)
inputs = dataSetX.iloc[:,:nn.kInputLayerSize].values
dataSetY = pandas.read_csv(sys.argv[2], sep=" ", header=None)
outputs = dataSetY.iloc[:,:nn.kOutputLayerSize].values

if kPrintDataSet:
  for i in range(len(inputs)):
    print("Sample #" + str(i))
    
    for j in range(nn.kInputLayerSize):
      print("  " + nn.kInputs[j] + ": " + str(inputs[i][j]))
      
    for j in range(nn.kOutputLayerSize):
      print("  -> " + nn.kOutputs[j] + ": " + str(outputs[i][j]))
      
    print("")
    
  input("Press Enter to continue...")

# Shuffle the data
inputs, outputs = shuffle(inputs, outputs)

# Split the data
inputsTrain, inputsTest, outputsTrain, outputsTest = train_test_split(inputs, outputs, test_size=kTestSetRatio)

# Create the neural network
network = nn.createNetwork()
print(network.summary())

trainingLearningRate = float(sys.argv[3])
opt = Adam(learning_rate=trainingLearningRate)
network.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])

# Train the neural network
trainingEpochs = int(sys.argv[4])
history = network.fit(inputsTrain, outputsTrain, validation_data=(inputsTest,outputsTest), epochs=trainingEpochs)

# Save the weights
if kSaveWeights:
  for layerIndex in range(len(network.layers)):
    weightsArray = network.layers[layerIndex].get_weights()[0]
    
    with open("./layer" + str(layerIndex) + "_weights", "w") as file:
      for i in range(len(weightsArray)):
        for j in range(len(weightsArray[i])):
          file.write(str(weightsArray[i][j]) + " ")
          
        file.write("\n")
        
    bias = network.layers[layerIndex].get_weights()[1]
    
    with open("./layer" + str(layerIndex) + "_bias", "w") as file:
      for i in range(len(bias)):
        file.write(str(bias[i]) + " ")
  
# Save the prediction
if kSavePrediction:
  prediction = network.predict(inputs)
  
  with open("./prediction_after_training", "w") as file:
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
