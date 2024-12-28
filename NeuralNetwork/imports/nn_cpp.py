
import imports.nn as nn

import os.path

kTab = "   "

def getEnumString(stringArray):
  enumString = ""

  for stringEntry in stringArray:
    if enumString != "":
      enumString += ",\n" + kTab + kTab + kTab
    
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

    weightsArrayString += "\n" + kTab + "{ "

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

    biasesArrayString += "\n" + kTab + str(biasesArray[i]) + "f"

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
  layerSizeDeclarations = kTab + "static const size_t kInputLayerSize = " + str(inputLayerSize) + "u;\n"
  layerMemberDeclarations = kTab + "float mInputs[kInputLayerSize];\n"

  staticLayerSetupDeclarations = ""

  dynamicLayerMemberDeclarations = ""
  dynamicLayerSetupDeclarations = ""

  hiddenLayerIndex = 0

  for hiddenLayer in hiddenLayers:
    hiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex)

    layerSizeDeclarations += kTab + "static const size_t k" + hiddenLayerString + "Size = " + str(hiddenLayer["HiddenLayerSize"]) + "u;\n"

    staticLayerSetupDeclarations += kTab + "static const float k" + hiddenLayerString + "Weights[k" + hiddenLayerString + "Size]"

    dynamicLayerMemberDeclarations += kTab + "float m" + hiddenLayerString + "Weights[k" + hiddenLayerString + "Size]"

    dynamicLayerSetupDeclarations += kTab + "inline float* get" + hiddenLayerString + "Weights()\n"
    dynamicLayerSetupDeclarations += kTab + "{\n"
    dynamicLayerSetupDeclarations += kTab + kTab + "return m" + hiddenLayerString + "Weights[0];\n"
    dynamicLayerSetupDeclarations += kTab + "}\n"
    dynamicLayerSetupDeclarations += kTab + "inline float* get" + hiddenLayerString + "Biases()\n"
    dynamicLayerSetupDeclarations += kTab + "{\n"
    dynamicLayerSetupDeclarations += kTab + kTab + "return m" + hiddenLayerString + "Biases;\n"
    dynamicLayerSetupDeclarations += kTab + "}\n"

    if hiddenLayerIndex == 0:
      staticLayerSetupDeclarations += "[kInputLayerSize];\n"
      dynamicLayerMemberDeclarations += "[kInputLayerSize];\n"
    else:
      previousHiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex - 1)      
      
      staticLayerSetupDeclarations += "[k" + previousHiddenLayerString + "Size];\n"
      dynamicLayerMemberDeclarations += "[k" + previousHiddenLayerString + "Size];\n"

    layerMemberDeclarations += kTab + "float m" + hiddenLayerString + "Values[k" + hiddenLayerString + "Size];\n"

    staticLayerSetupDeclarations += kTab + "static const float k" + hiddenLayerString + "Biases[k" + hiddenLayerString + "Size];\n"
    staticLayerSetupDeclarations += kTab + "static float (*k" + hiddenLayerString + "Activation)(float);\n\n"

    dynamicLayerMemberDeclarations += kTab + "float m" + hiddenLayerString + "Biases[k" + hiddenLayerString + "Size];\n"
    dynamicLayerMemberDeclarations += kTab + "float (*m" + hiddenLayerString + "Activation)(float);\n\n"

    hiddenLayerIndex = hiddenLayerIndex + 1

  layerSizeDeclarations += kTab + "static const size_t kOutputLayerSize = " + str(outputLayerSize) + "u;"
  layerMemberDeclarations += kTab + "float mOutputs[kOutputLayerSize];"

  staticLayerSetupDeclarations += kTab + "static const float kOutputLayerWeights[kOutputLayerSize][kHiddenLayer" + str(hiddenLayersCount - 1) + "Size];\n"
  staticLayerSetupDeclarations += kTab + "static const float kOutputLayerBiases[kOutputLayerSize];\n"
  staticLayerSetupDeclarations += kTab + "static float (*kOutputLayerActivation)(float);"

  dynamicLayerMemberDeclarations += kTab + "float mOutputLayerWeights[kOutputLayerSize][k" + hiddenLayerString + "Size];\n"
  dynamicLayerMemberDeclarations += kTab + "float mOutputLayerBiases[kOutputLayerSize];\n"
  dynamicLayerMemberDeclarations += kTab + "float (*mOutputLayerActivation)(float);"

  dynamicLayerSetupDeclarations += kTab + "inline float* getOutputLayerWeights()\n"
  dynamicLayerSetupDeclarations += kTab + "{\n"
  dynamicLayerSetupDeclarations += kTab + kTab + "return mOutputLayerWeights[0];\n"
  dynamicLayerSetupDeclarations += kTab + "}\n"
  dynamicLayerSetupDeclarations += kTab + "inline float* getOutputLayerBiases()\n"
  dynamicLayerSetupDeclarations += kTab + "{\n"
  dynamicLayerSetupDeclarations += kTab + kTab + "return mOutputLayerBiases;\n"
  dynamicLayerSetupDeclarations += kTab + "}"

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)  
  generatedContent = generatedContent.replace("$LayerSizeDeclarations$", layerSizeDeclarations)
  generatedContent = generatedContent.replace("$LayerMemberDeclarations$", layerMemberDeclarations)
  generatedContent = generatedContent.replace("$StaticLayerSetupDeclarations$", staticLayerSetupDeclarations)
  generatedContent = generatedContent.replace("$DynamicLayerMemberDeclarations$", dynamicLayerMemberDeclarations)
  generatedContent = generatedContent.replace("$DynamicLayerSetupDeclarations$", dynamicLayerSetupDeclarations)

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
  layerValuesInitialization = kTab + "memset(mInputs, 0, sizeof(mInputs));\n"
  predictionCode = ""
  staticLayerSetupDefinitions = ""
  dynamicLayerSetupInitialization = ""
  dynamicLoadSetupCode = kTab + "char filePath[kFilePathMaxSize];\n"
  dynamicLoadSetupCode += kTab + "std::ifstream file;\n\n"

  hiddenLayerIndex = 0

  for hiddenLayer in hiddenLayers:
    hiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex)
    hiddenIndexVariable = "hidden" + str(hiddenLayerIndex) + "Index"

    layerValuesInitialization += kTab + "memset(m" + hiddenLayerString + "Values, "
    layerValuesInitialization += "0, sizeof(m" + hiddenLayerString + "Values));\n"

    predictionCode += kTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString
    predictionCode += "Size; " + hiddenIndexVariable + "++)\n"
    predictionCode += kTab + "{\n"
    predictionCode += kTab + kTab + "float sum = $Prefix$" + hiddenLayerString + "Biases[" + hiddenIndexVariable + "];\n\n"

    staticLayerSetupDefinitions += "const float " + name + "::k" + hiddenLayerString + "Weights[k" + hiddenLayerString + "Size]"

    dynamicLayerSetupInitialization += kTab + "memset(m" + hiddenLayerString + "Weights, 0, sizeof(m" + hiddenLayerString + "Weights));\n"
    dynamicLayerSetupInitialization += kTab + "memset(m" + hiddenLayerString + "Biases, 0, sizeof(m" + hiddenLayerString + "Biases));\n"
    dynamicLayerSetupInitialization += kTab + "m" + hiddenLayerString + "Activation = activation" + hiddenLayer["HiddenLayerActivation"] + ";\n\n"

    dynamicLoadSetupCode += kTab + "snprintf(filePath, kFilePathMaxSize, \"%s_layer" + str(hiddenLayerIndex) + "_weights_\", pDataDirectory);\n"
    dynamicLoadSetupCode += kTab + "file = std::ifstream(filePath);\n\n"
    dynamicLoadSetupCode += kTab + "if(file.is_open())\n"
    dynamicLoadSetupCode += kTab + "{\n"

    if hiddenLayerIndex == 0:
      predictionCode += kTab + kTab + "for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)\n"
      predictionCode += kTab + kTab + "{\n"
      predictionCode += kTab + kTab + kTab + "sum += mInputs[inputIndex] * $Prefix$" + hiddenLayerString + "Weights[" + hiddenIndexVariable + "][inputIndex];\n"
      predictionCode += kTab + kTab + "}\n\n"

      staticLayerSetupDefinitions += "[kInputLayerSize] =\n"

      dynamicLoadSetupCode += kTab + kTab + "for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)\n"
      dynamicLoadSetupCode += kTab + kTab + "{\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString + "Size; " + hiddenIndexVariable + "++)\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "{\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + kTab + "file >> m" + hiddenLayerString + "Weights[" + hiddenIndexVariable + "][inputIndex];\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "}\n"
      dynamicLoadSetupCode += kTab + kTab + "}\n\n"
    else:
      previousHiddenLayerString = "HiddenLayer" + str(hiddenLayerIndex - 1)
      previousHiddenIndexVariable = "hidden" + str(hiddenLayerIndex - 1) + "Index"

      predictionCode += kTab + kTab + "for(size_t " + previousHiddenIndexVariable + " = 0u; "
      predictionCode += previousHiddenIndexVariable + " < k" + previousHiddenLayerString + "Size; " + previousHiddenIndexVariable + "++)\n"
      predictionCode += kTab + kTab + "{\n"
      predictionCode += kTab + kTab + kTab + "sum += m" + previousHiddenLayerString + "Values[" + previousHiddenIndexVariable + "] * $Prefix$"
      predictionCode += hiddenLayerString + "Weights[" + hiddenIndexVariable + "][" + previousHiddenIndexVariable + "];\n"
      predictionCode += kTab + kTab + "}\n\n"

      staticLayerSetupDefinitions += "[k" + previousHiddenLayerString + "Size] =\n"

      dynamicLoadSetupCode += kTab + kTab + "for(size_t " + previousHiddenIndexVariable + " = 0u; " + previousHiddenIndexVariable + " < k" + previousHiddenLayerString + "Size; " + previousHiddenIndexVariable + "++)\n"
      dynamicLoadSetupCode += kTab + kTab + "{\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString + "Size; " + hiddenIndexVariable + "++)\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "{\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + kTab + "file >> m" + hiddenLayerString + "Weights[" + hiddenIndexVariable + "][" + previousHiddenIndexVariable + "];\n"
      dynamicLoadSetupCode += kTab + kTab + kTab + "}\n"
      dynamicLoadSetupCode += kTab + kTab + "}\n\n"

    predictionCode += kTab + kTab + "m" + hiddenLayerString + "Values[" + hiddenIndexVariable + "] = $Prefix$" + hiddenLayerString + "Activation(sum);\n"
    predictionCode += kTab + "}\n\n"

    staticLayerSetupDefinitions += getWeightsArrayString(network, hiddenLayerIndex) + ";\n"
    staticLayerSetupDefinitions += "const float " + name + "::k" + hiddenLayerString + "Biases[k" + hiddenLayerString + "Size] =\n"
    staticLayerSetupDefinitions += getBiasesArrayString(network, hiddenLayerIndex) + ";\n"
    staticLayerSetupDefinitions += "float (*" + name + "::k" + hiddenLayerString + "Activation)(float) = "
    staticLayerSetupDefinitions += "activation" + hiddenLayer["HiddenLayerActivation"] + ";\n\n"

    dynamicLoadSetupCode += kTab + kTab + "file.close();\n"
    dynamicLoadSetupCode += kTab + "}\n\n"
    dynamicLoadSetupCode += kTab + "snprintf(filePath, kFilePathMaxSize, \"%s_layer" + str(hiddenLayerIndex) + "_biases_\", pDataDirectory);\n"
    dynamicLoadSetupCode += kTab + "file = std::ifstream(filePath);\n\n"
    dynamicLoadSetupCode += kTab + "if(file.is_open())\n"
    dynamicLoadSetupCode += kTab + "{\n"
    dynamicLoadSetupCode += kTab + kTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString + "Size; " + hiddenIndexVariable + "++)\n"
    dynamicLoadSetupCode += kTab + kTab + "{\n"
    dynamicLoadSetupCode += kTab + kTab + kTab + "file >> m" + hiddenLayerString + "Biases[" + hiddenIndexVariable + "];\n"
    dynamicLoadSetupCode += kTab + kTab + "}\n\n"
    dynamicLoadSetupCode += kTab + kTab + "file.close();\n"
    dynamicLoadSetupCode += kTab + "}\n\n"

    hiddenLayerIndex = hiddenLayerIndex + 1

  layerValuesInitialization += kTab + "memset(mOutputs, 0, sizeof(mOutputs));"

  predictionCode += kTab + "for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)\n"
  predictionCode += kTab + "{\n"
  predictionCode += kTab + kTab + "float sum = $Prefix$OutputLayerBiases[outputIndex];\n\n"
  predictionCode += kTab + kTab + "for(size_t " + hiddenIndexVariable + " = 0u; "
  predictionCode += hiddenIndexVariable + " < k" + hiddenLayerString + "Size; " + hiddenIndexVariable + "++)\n"
  predictionCode += kTab + kTab + "{\n"
  predictionCode += kTab + kTab + kTab + "sum += m" + hiddenLayerString + "Values[" + hiddenIndexVariable
  predictionCode += "] * $Prefix$OutputLayerWeights[outputIndex][" + hiddenIndexVariable + "];\n"
  predictionCode += kTab + kTab + "}\n\n"
  predictionCode += kTab + kTab + "mOutputs[outputIndex] = $Prefix$OutputLayerActivation(sum);\n"
  predictionCode += kTab + "}"

  outputLayerIndex = len(hiddenLayers)

  staticLayerSetupDefinitions += "const float " + name + "::kOutputLayerWeights[kOutputLayerSize][k" + hiddenLayerString + "Size] =\n"
  staticLayerSetupDefinitions += getWeightsArrayString(network, outputLayerIndex) + ";\n"
  staticLayerSetupDefinitions += "const float " + name + "::kOutputLayerBiases[kOutputLayerSize] =\n"
  staticLayerSetupDefinitions += getBiasesArrayString(network, outputLayerIndex) + ";"

  dynamicLayerSetupInitialization += kTab + "memset(mOutputLayerWeights, 0, sizeof(mOutputLayerWeights));\n"
  dynamicLayerSetupInitialization += kTab + "memset(mOutputLayerBiases, 0, sizeof(mOutputLayerBiases));\n"
  dynamicLayerSetupInitialization += kTab + "mOutputLayerActivation = activation" + outputLayerActivation + ";"

  dynamicLoadSetupCode += kTab + "snprintf(filePath, kFilePathMaxSize, \"%s_layer" + str(outputLayerIndex) + "_weights_\", pDataDirectory);\n"
  dynamicLoadSetupCode += kTab + "file = std::ifstream(filePath);\n\n"
  dynamicLoadSetupCode += kTab + "if(file.is_open())\n"
  dynamicLoadSetupCode += kTab + "{\n"
  dynamicLoadSetupCode += kTab + kTab + "for(size_t " + hiddenIndexVariable + " = 0u; " + hiddenIndexVariable + " < k" + hiddenLayerString + "Size; " + hiddenIndexVariable + "++)\n"
  dynamicLoadSetupCode += kTab + kTab + "{\n"
  dynamicLoadSetupCode += kTab + kTab + kTab + "for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)\n"
  dynamicLoadSetupCode += kTab + kTab + kTab + "{\n"
  dynamicLoadSetupCode += kTab + kTab + kTab + kTab + "file >> mOutputLayerWeights[outputIndex][" + hiddenIndexVariable + "];\n"
  dynamicLoadSetupCode += kTab + kTab + kTab + "}\n"
  dynamicLoadSetupCode += kTab + kTab + "}\n\n"
  dynamicLoadSetupCode += kTab + kTab + "file.close();\n"
  dynamicLoadSetupCode += kTab + "}\n\n"
  dynamicLoadSetupCode += kTab + "snprintf(filePath, kFilePathMaxSize, \"%s_layer" + str(outputLayerIndex) + "_biases_\", pDataDirectory);\n"
  dynamicLoadSetupCode += kTab + "file = std::ifstream(filePath);\n\n"
  dynamicLoadSetupCode += kTab + "if(file.is_open())\n"
  dynamicLoadSetupCode += kTab + "{\n"
  dynamicLoadSetupCode += kTab + kTab + "for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)\n"
  dynamicLoadSetupCode += kTab + kTab + "{\n"
  dynamicLoadSetupCode += kTab + kTab + kTab + "file >> mOutputLayerBiases[outputIndex];\n"
  dynamicLoadSetupCode += kTab + kTab + "}\n\n"
  dynamicLoadSetupCode += kTab + kTab + "file.close();\n"
  dynamicLoadSetupCode += kTab + "}"

  # Replace variables
  generatedContent = templateContent.replace("$Name$", name)
  generatedContent = generatedContent.replace("$LeakyReLUAlpha$", str(leakyReLUAlpha))
  generatedContent = generatedContent.replace("$LayerValuesInitialization$", layerValuesInitialization)
  generatedContent = generatedContent.replace("$StaticPredictionCode$", predictionCode.replace("$Prefix$", "k"))
  generatedContent = generatedContent.replace("$StaticLayerSetupDefinitions$", staticLayerSetupDefinitions)
  generatedContent = generatedContent.replace("$DynamicPredictionCode$", predictionCode.replace("$Prefix$", "m"))
  generatedContent = generatedContent.replace("$DynamicLayerSetupInitialization$", dynamicLayerSetupInitialization)
  generatedContent = generatedContent.replace("$DynamicLoadSetupCode$", dynamicLoadSetupCode)

  # Generate source file
  generatedFilePath = os.path.join(dataDirectory, name + ".cpp")
  generatedFile = open(generatedFilePath, "w")
  generatedFile.write(generatedContent)
  generatedFile.close()
