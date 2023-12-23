
#include "$Name$.h"

#include <cmath>
#include <cstdio>
#include <fstream>

float $Name$::activationSigmoid(float pValue)
{
   return 1.0f / (1.0f + expf(-pValue));
}
float $Name$::activationReLU(float pValue)
{
   return pValue > 0.0f ? pValue : 0.0f;
}
float $Name$::activationLeakyReLU(float pValue)
{
   static const float kAlpha = 0.01f;
   return pValue > 0.0f ? pValue : (pValue * kAlpha);
}

$Name$::$Name$()
   : mHiddenLayerActivation(activationLeakyReLU)
   , mOutputLayerActivation(activationLeakyReLU)
{
   memset(mInputs, 0, sizeof(mInputs));
   memset(mHiddenLayerValues, 0, sizeof(mHiddenLayerValues));
   memset(mOutputs, 0, sizeof(mOutputs));
}

void $Name$::predict()
{
   for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
   {
      float sum = kHiddenLayerBiases[hiddenIndex];

      for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)
      {
         sum += mInputs[inputIndex] * kHiddenLayerWeights[hiddenIndex][inputIndex];
      }

      mHiddenLayerValues[hiddenIndex] = mHiddenLayerActivation(sum);
   }

   for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)
   {
      float sum = kOutputLayerBiases[outputIndex];

      for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
      {
         sum += mHiddenLayerValues[hiddenIndex] * kOutputLayerWeights[outputIndex][hiddenIndex];
      }

      mOutputs[outputIndex] = mOutputLayerActivation(sum);
   }
}

void $Name$::test(const char* pInputsDirectory)
{
   static const size_t kFilePathMaxSize = 256u;

   char inputsFilePath[kFilePathMaxSize];
   snprintf(inputsFilePath, kFilePathMaxSize, "%s_inputs_", pInputsDirectory);

   std::ifstream inputsFile(inputsFilePath);
   
   if(!inputsFile.is_open())
   {
      return;
   }

   char outputsFilePath[kFilePathMaxSize];
   snprintf(outputsFilePath, kFilePathMaxSize, "%s_prediction_cpp_", pInputsDirectory);

   std::ofstream predictionFile(outputsFilePath);

   if(!predictionFile.is_open())
   {
      inputsFile.close();
      return;
   }

   predictionFile << std::fixed << std::showpoint;

   while(!inputsFile.eof())
   {
      for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)
      {
         inputsFile >> mInputs[inputIndex];
      }

      predict();

      for(size_t outputIndex = 0; outputIndex < kOutputLayerSize; outputIndex++)
      {
         predictionFile << mOutputs[outputIndex] << " ";
      }

      predictionFile << "\n";
   }

   predictionFile.close();
   inputsFile.close();
}

const float $Name$::kHiddenLayerWeights[kHiddenLayerSize][kInputLayerSize] =
$HiddenLayerWeights$;
const float $Name$::kHiddenLayerBiases[kHiddenLayerSize] =
$HiddenLayerBiases$;

const float $Name$::kOutputLayerWeights[kOutputLayerSize][kHiddenLayerSize] =
$OutputLayerWeights$;
const float $Name$::kOutputLayerBiases[kOutputLayerSize] =
$OutputLayerBiases$;
