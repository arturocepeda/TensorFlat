
import nn
import sys
import json


def getEnumString(stringArray):
  enumString = ""

  for stringEntry in stringArray:
    if enumString != "":
      enumString += ",\n         "
    
    enumString += "k" + stringEntry

  return enumString


def getWeightsArrayString(network, layerIndex):
  weightsArrayString = "{"
  weightsArray = network.layers[layerIndex].get_weights()[0]

  inputCount = len(weightsArray[0])
  outputCount = len(weightsArray)

  for i in range(inputCount):
    if i > 0:
      weightsArrayString += ","

    weightsArrayString += "\n   { "

    for j in range(outputCount):
      if j > 0:
        weightsArrayString += ", "

      weightsArrayString += str(weightsArray[j][i]) + "f"

    weightsArrayString += " }"

  weightsArrayString += "\n}"
  return weightsArrayString


def getBiasesArrayString(network, layerIndex):
  biasesArrayString = "{"
  biasesArray = network.layers[layerIndex].get_weights()[1]

  for i in range(len(biasesArray)):
    if i > 0:
      biasesArrayString += ", "

    biasesArrayString += "\n   " + str(biasesArray[i]) + "f"

  biasesArrayString += "\n}"
  return biasesArrayString


if __name__ == "__main__":
  # Argument check
  if len(sys.argv) < 2:
    print("Usage: nn_generate_cpp.py <name>")
    exit(1)

  name = sys.argv[1]
  network = nn.createNetwork(name)

  # Load data from the description file
  jsonFilePath = name + "/nn.json"
  jsonFile = open(jsonFilePath)
  jsonData = json.load(jsonFile)

  inputs = jsonData["Inputs"]
  hiddenLayerSize = jsonData["HiddenLayerSize"]
  outputs = jsonData["Outputs"]

  jsonFile.close()

  inputLayerSize = len(inputs)
  outputLayerSize = len(outputs)

  # Load header template
  templateFilePath = "./templates/nn_cpp_template.h"
  templateFile = open(templateFilePath, "r")
  templateContent = templateFile.read()
  templateFile.close()

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)
  generatedContent = generatedContent.replace("$InputLayerSize$", str(inputLayerSize) + "u")
  generatedContent = generatedContent.replace("$HiddenLayerSize$", str(hiddenLayerSize) + "u")
  generatedContent = generatedContent.replace("$OutputLayerSize$", str(outputLayerSize) + "u")

  inputsEnumString = getEnumString(inputs)
  generatedContent = generatedContent.replace("$InputsEnum$", inputsEnumString)

  outputsEnumString = getEnumString(outputs)
  generatedContent = generatedContent.replace("$OutputsEnum$", outputsEnumString)

  # Generate header file
  generatedFilePath = name + "/" + name + ".h"
  generatedFile = open(generatedFilePath, "w")
  generatedFile.write(generatedContent)
  generatedFile.close()

  # Load source template
  templateFilePath = "./templates/nn_cpp_template.cpp"
  templateFile = open(templateFilePath, "r")
  templateContent = templateFile.read()
  templateFile.close()

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)

  hiddenLayerWeightsString = getWeightsArrayString(network, 0)
  generatedContent = generatedContent.replace("$HiddenLayerWeights$", hiddenLayerWeightsString)
  hiddenLayerBiasesString = getBiasesArrayString(network, 0)
  generatedContent = generatedContent.replace("$HiddenLayerBiases$", hiddenLayerBiasesString)

  outputLayerWeightsString = getWeightsArrayString(network, 1)
  generatedContent = generatedContent.replace("$OutputLayerWeights$", outputLayerWeightsString)
  outputLayerBiasesString = getBiasesArrayString(network, 1)
  generatedContent = generatedContent.replace("$OutputLayerBiases$", outputLayerBiasesString)

  # Generate source file
  generatedFilePath = name + "/" + name + ".cpp"
  generatedFile = open(generatedFilePath, "w")
  generatedFile.write(generatedContent)
  generatedFile.close()
