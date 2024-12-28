
import imports.nn as nn

import os.path

kCppTab = "   "

def getEnumString(stringArray):
  enumString = ""

  for stringEntry in stringArray:
    if enumString != "":
      enumString += ",\n" + kCppTab + kCppTab + kCppTab
    
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

    weightsArrayString += "\n" + kCppTab + "{ "

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

    biasesArrayString += "\n" + kCppTab + str(biasesArray[i]) + "f"

  biasesArrayString += "\n}"
  return biasesArrayString


def generateHeaderFile(dataDirectory, templateFileName):
  descriptionData = nn.loadNetworkDescriptionData(dataDirectory)

  name = descriptionData["Name"]
  inputs = descriptionData["Inputs"]
  hiddenLayers = descriptionData["HiddenLayers"]
  outputs = descriptionData["Outputs"]

  inputLayerSize = len(inputs)
  hiddenLayersCount = len(hiddenLayers)
  outputLayerSize = len(outputs)

  # Load header template
  templateFilePath = "./templates/" + templateFileName
  templateFile = open(templateFilePath, "r")
  templateContent = templateFile.read()
  templateFile.close()

  # Generate required code chunks
  layerSizeDeclarations = kCppTab + "static const size_t kInputLayerSize = " + str(inputLayerSize) + "u;\n"
  layerSetupDeclarations = ""
  layerMemberDeclarations = kCppTab + "float mInputs[kInputLayerSize];\n"

  hiddenLayerIndex = 0

  for hiddenLayer in hiddenLayers:
    hiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex)

    layerSizeDeclarations += kCppTab + "static const size_t k" + hiddenLayerString + "Size = " + str(hiddenLayer["HiddenLayerSize"]) + "u;\n"
    layerSetupDeclarations += kCppTab + "static const float k" + hiddenLayerString + "Weights[k" + hiddenLayerString + "Size]"

    if hiddenLayerIndex == 0:
      layerSetupDeclarations += "[kInputLayerSize];\n"
    else:
      previousHiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex - 1)      
      
      layerSetupDeclarations += "[k" + previousHiddenLayerString + "Size];\n"

    layerSetupDeclarations += kCppTab + "static const float k" + hiddenLayerString + "Biases[k" + hiddenLayerString + "Size];\n"
    layerSetupDeclarations += kCppTab + "static float (*k" + hiddenLayerString + "Activation)(float);\n\n"

    layerMemberDeclarations += kCppTab + "float m" + hiddenLayerString + "Values[k" + hiddenLayerString + "Size];\n"

    hiddenLayerIndex = hiddenLayerIndex + 1

  layerSizeDeclarations += kCppTab + "static const size_t kOutputLayerSize = " + str(outputLayerSize) + "u;"
  layerSetupDeclarations += kCppTab + "static const float kOutputLayerWeights[kOutputLayerSize][kHiddenLayer" + str(hiddenLayersCount - 1) + "Size];\n"
  layerSetupDeclarations += kCppTab + "static const float kOutputLayerBiases[kOutputLayerSize];\n"
  layerSetupDeclarations += kCppTab + "static float (*kOutputLayerActivation)(float);"
  layerMemberDeclarations += kCppTab + "float mOutputs[kOutputLayerSize];"

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)
  generatedContent = generatedContent.replace("$LayerSizeDeclarations$", layerSizeDeclarations)
  generatedContent = generatedContent.replace("$LayerSetupDeclarations$", layerSetupDeclarations)
  generatedContent = generatedContent.replace("$LayerMemberDeclarations$", layerMemberDeclarations)

  inputsEnumString = getEnumString(inputs)
  generatedContent = generatedContent.replace("$InputsEnum$", inputsEnumString)

  outputsEnumString = getEnumString(outputs)
  generatedContent = generatedContent.replace("$OutputsEnum$", outputsEnumString)

  # Generate header file
  generatedFilePath = os.path.join(dataDirectory, name + ".h")
  generatedFile = open(generatedFilePath, "w")
  generatedFile.write(generatedContent)
  generatedFile.close()


def generateSourceFile(dataDirectory, templateFileName):
  descriptionData = nn.loadNetworkDescriptionData(dataDirectory)
  
  name = descriptionData["Name"]
  hiddenLayers = descriptionData["HiddenLayers"]
  leakyReLUAlpha = descriptionData["LeakyReLUAlpha"]
  outputLayerActivation = descriptionData["OutputLayerActivation"]

  network = nn.createNetwork(dataDirectory)

  # Load source template
  templateFilePath = "./templates/" + templateFileName
  templateFile = open(templateFilePath, "r")
  templateContent = templateFile.read()
  templateFile.close()

  # Generate required code chunks
  layerValuesInitialization = kCppTab + "memset(mInputs, 0, sizeof(mInputs));\n"
  predictionCode = ""
  layerSetupDefinitions = ""

  hiddenLayerIndex = 0

  for hiddenLayer in hiddenLayers:
    hiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex)
    hiddenIndexVariable = "hidden" + str(hiddenLayerIndex) + "Index"

    layerValuesInitialization += kCppTab + "memset(m" + hiddenLayerString + "Values, "
    layerValuesInitialization += "0, sizeof(m" + hiddenLayerString + "Values));\n"

    predictionCode += kCppTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString
    predictionCode += "Size; " + hiddenIndexVariable + "++)\n"
    predictionCode += kCppTab + "{\n"
    predictionCode += kCppTab + kCppTab + "float sum = k" + hiddenLayerString + "Biases[" + hiddenIndexVariable + "];\n\n"

    layerSetupDefinitions += "const float " + name + "::k" + hiddenLayerString + "Weights[k" + hiddenLayerString + "Size]"

    if hiddenLayerIndex == 0:
      predictionCode += kCppTab + kCppTab + "for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)\n"
      predictionCode += kCppTab + kCppTab + "{\n"
      predictionCode += kCppTab + kCppTab + kCppTab + "sum += mInputs[inputIndex] * k" + hiddenLayerString + "Weights[" + hiddenIndexVariable + "][inputIndex];\n"
      predictionCode += kCppTab + kCppTab + "}\n\n"

      layerSetupDefinitions += "[kInputLayerSize] =\n"
    else:
      previousHiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex - 1)
      previousHiddenIndexVariable = "hidden" + str(hiddenLayerIndex - 1) + "Index"

      predictionCode += kCppTab + kCppTab + "for(size_t " + previousHiddenIndexVariable + " = 0u; "
      predictionCode += previousHiddenIndexVariable + " < k" + previousHiddenLayerString + "Size; " + previousHiddenIndexVariable + "++)\n"
      predictionCode += kCppTab + kCppTab + "{\n"
      predictionCode += kCppTab + kCppTab + kCppTab + "sum += m" + previousHiddenLayerString + "Values[" + previousHiddenIndexVariable + "] * k"
      predictionCode += hiddenLayerString + "Weights[" + hiddenIndexVariable + "][" + previousHiddenIndexVariable + "];\n"
      predictionCode += kCppTab + kCppTab + "}\n\n"

      layerSetupDefinitions += "[k" + previousHiddenLayerString + "Size] =\n"

    predictionCode += kCppTab + kCppTab + "m" + hiddenLayerString + "Values[" + hiddenIndexVariable + "] = k" + hiddenLayerString + "Activation(sum);\n"
    predictionCode += kCppTab + "}\n\n"

    layerSetupDefinitions += getWeightsArrayString(network, hiddenLayerIndex) + ";\n"
    layerSetupDefinitions += "const float " + name + "::k" + hiddenLayerString + "Biases[k" + hiddenLayerString + "Size] =\n"
    layerSetupDefinitions += getBiasesArrayString(network, hiddenLayerIndex) + ";\n"
    layerSetupDefinitions += "float (*" + name + "::k" + hiddenLayerString + "Activation)(float) = "
    layerSetupDefinitions += "activation" + hiddenLayer["HiddenLayerActivation"] + ";\n\n"

    hiddenLayerIndex = hiddenLayerIndex + 1

  layerValuesInitialization += kCppTab + "memset(mOutputs, 0, sizeof(mOutputs));"

  lastHiddenLayerIndex = len(hiddenLayers) - 1
  lastHiddenLayerString = "HiddenLayer" + str(lastHiddenLayerIndex)
  lastHiddenIndexVariable = "hidden" + str(lastHiddenLayerIndex) + "Index"

  predictionCode += kCppTab + "for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)\n"
  predictionCode += kCppTab + "{\n"
  predictionCode += kCppTab + kCppTab + "float sum = kOutputLayerBiases[outputIndex];\n\n"
  predictionCode += kCppTab + kCppTab + "for(size_t " + lastHiddenIndexVariable + " = 0u; "
  predictionCode += lastHiddenIndexVariable + " < k" + lastHiddenLayerString + "Size; " + lastHiddenIndexVariable + "++)\n"
  predictionCode += kCppTab + kCppTab + "{\n"
  predictionCode += kCppTab + kCppTab + kCppTab + "sum += m" + lastHiddenLayerString + "Values[" + lastHiddenIndexVariable
  predictionCode += "] * kOutputLayerWeights[outputIndex][" + lastHiddenIndexVariable + "];\n"
  predictionCode += kCppTab + kCppTab + "}\n\n"
  predictionCode += kCppTab + kCppTab + "mOutputs[outputIndex] = kOutputLayerActivation(sum);\n"
  predictionCode += kCppTab + "}"

  outputLayerIndex = len(hiddenLayers)

  layerSetupDefinitions += "const float " + name + "::kOutputLayerWeights[kOutputLayerSize][k" + lastHiddenLayerString + "Size] =\n"
  layerSetupDefinitions += getWeightsArrayString(network, outputLayerIndex) + ";\n"
  layerSetupDefinitions += "const float " + name + "::kOutputLayerBiases[kOutputLayerSize] =\n"
  layerSetupDefinitions += getBiasesArrayString(network, outputLayerIndex) + ";"

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)
  generatedContent = generatedContent.replace("$LeakyReLUAlpha$", str(leakyReLUAlpha))
  generatedContent = generatedContent.replace("$OutputLayerActivation$", outputLayerActivation)
  generatedContent = generatedContent.replace("$LayerValuesInitialization$", layerValuesInitialization)
  generatedContent = generatedContent.replace("$PredictionCode$", predictionCode)
  generatedContent = generatedContent.replace("$LayerSetupDefinitions$", layerSetupDefinitions)

  # Generate source file
  generatedFilePath = os.path.join(dataDirectory, name + ".cpp")
  generatedFile = open(generatedFilePath, "w")
  generatedFile.write(generatedContent)
  generatedFile.close()
