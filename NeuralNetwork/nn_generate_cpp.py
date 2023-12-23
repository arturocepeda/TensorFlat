
import sys
import json

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_generate_cpp.py <name>")
  exit(1)

name = sys.argv[1]

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

inputsEnumString = ""

for input in inputs:
  if inputsEnumString != "":
    inputsEnumString += ",\n      "
  
  inputsEnumString += "k" + input

generatedContent = generatedContent.replace("$InputsEnum$", inputsEnumString)

outputsEnumString = ""

for output in outputs:
  if outputsEnumString != "":
    outputsEnumString += ",\n      "
  
  outputsEnumString += "k" + output

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
generatedContent = generatedContent.replace("$InputLayerSize$", str(inputLayerSize) + "u")
generatedContent = generatedContent.replace("$HiddenLayerSize$", str(hiddenLayerSize) + "u")
generatedContent = generatedContent.replace("$OutputLayerSize$", str(outputLayerSize) + "u")

hiddenLayerWeights = "{"
hiddenLayerWeights += "\n   0.0f" #TODO
hiddenLayerWeights += "\n}"

generatedContent = generatedContent.replace("$HiddenLayerWeights$", hiddenLayerWeights)

hiddenLayerBiases = "{"
hiddenLayerBiases += "\n   0.0f" #TODO
hiddenLayerBiases += "\n}"

generatedContent = generatedContent.replace("$HiddenLayerBiases$", hiddenLayerBiases)

outputLayerWeights = "{"
outputLayerWeights += "\n   0.0f" #TODO
outputLayerWeights += "\n}"

generatedContent = generatedContent.replace("$OutputLayerWeights$", outputLayerWeights)

outputLayerBiases = "{"
outputLayerBiases += "\n   0.0f" #TODO
outputLayerBiases += "\n}"

generatedContent = generatedContent.replace("$OutputLayerBiases$", outputLayerBiases)

# Generate source file
generatedFilePath = name + "/" + name + ".cpp"
generatedFile = open(generatedFilePath, "w")
generatedFile.write(generatedContent)
generatedFile.close()
