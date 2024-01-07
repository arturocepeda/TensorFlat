
#include "$Name$.h"

#include <cmath>
#include <cstdio>
#include <iomanip>


static const size_t kFilePathMaxSize = 256u;
static const int kFileFloatPrecision = 5;


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
   static const float kAlpha = $LeakyReLUAlpha$f;
   return pValue > 0.0f ? pValue : (pValue * kAlpha);
}


$Name$::$Name$()
   : mHiddenLayerActivation(activation$HiddenLayerActivation$)
   , mOutputLayerActivation(activation$OutputLayerActivation$)
{
   memset(mHiddenLayerWeights, 0, sizeof(mHiddenLayerWeights));
   memset(mHiddenLayerBiases, 0, sizeof(mHiddenLayerBiases));

   memset(mOutputLayerWeights, 0, sizeof(mOutputLayerWeights));
   memset(mOutputLayerBiases, 0, sizeof(mOutputLayerBiases));

   memset(mInputs, 0, sizeof(mInputs));
   memset(mHiddenLayerValues, 0, sizeof(mHiddenLayerValues));
   memset(mOutputs, 0, sizeof(mOutputs));
}

void $Name$::loadWeightsAndBiases(const char* pDataDirectory)
{
   char filePath[kFilePathMaxSize];
   snprintf(filePath, kFilePathMaxSize, "%s_layer0_weights_", pDataDirectory);

   std::ifstream file(filePath);

   if(file.is_open())
   {
      for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)
      {
         for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
         {
            file >> mHiddenLayerWeights[hiddenIndex][inputIndex];
         }
      }

      file.close();
   }

   snprintf(filePath, kFilePathMaxSize, "%s_layer0_biases_", pDataDirectory);   
   file = std::ifstream(filePath);

   if(file.is_open())
   {
      for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
      {
         file >> mHiddenLayerBiases[hiddenIndex];
      }

      file.close();
   }

   snprintf(filePath, kFilePathMaxSize, "%s_layer1_weights_", pDataDirectory);
   file = std::ifstream(filePath);

   if(file.is_open())
   {
      for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
      {
         for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)
         {
            file >> mOutputLayerWeights[outputIndex][hiddenIndex];
         }
      }

      file.close();
   }

   snprintf(filePath, kFilePathMaxSize, "%s_layer1_biases_", pDataDirectory);   
   file = std::ifstream(filePath);

   if(file.is_open())
   {
      for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)
      {
         file >> mOutputLayerBiases[outputIndex];
      }

      file.close();
   }
}

void $Name$::predict()
{
   for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
   {
      float sum = mHiddenLayerBiases[hiddenIndex];

      for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)
      {
         sum += mInputs[inputIndex] * mHiddenLayerWeights[hiddenIndex][inputIndex];
      }

      mHiddenLayerValues[hiddenIndex] = mHiddenLayerActivation(sum);
   }

   for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)
   {
      float sum = mOutputLayerBiases[outputIndex];

      for(size_t hiddenIndex = 0u; hiddenIndex < kHiddenLayerSize; hiddenIndex++)
      {
         sum += mHiddenLayerValues[hiddenIndex] * mOutputLayerWeights[outputIndex][hiddenIndex];
      }

      mOutputs[outputIndex] = mOutputLayerActivation(sum);
   }
}

void $Name$::captureStart(const char* pDataDirectory)
{
   char inputsFilePath[kFilePathMaxSize];
   snprintf(inputsFilePath, kFilePathMaxSize, "%s_inputs_", pDataDirectory);

   mInputsStream = std::ofstream(inputsFilePath, std::ios_base::app);

   if(mInputsStream.is_open())
   {
      mInputsStream << std::fixed << std::showpoint << std::setprecision(kFileFloatPrecision);
   }

   char outputsFilePath[kFilePathMaxSize];
   snprintf(outputsFilePath, kFilePathMaxSize, "%s_outputs_", pDataDirectory);

   mOutputsStream = std::ofstream(outputsFilePath, std::ios_base::app);

   if(mOutputsStream.is_open())
   {
      mOutputsStream << std::fixed << std::showpoint << std::setprecision(kFileFloatPrecision);
   }
}

void $Name$::captureSample()
{
   if(mInputsStream.is_open())
   {
      for(size_t inputIndex = 0u; inputIndex < kInputLayerSize; inputIndex++)
      {
         mInputsStream << mInputs[inputIndex] << " ";
      }

      mInputsStream << "\n";
   }

   if(mOutputsStream.is_open())
   {
      for(size_t outputIndex = 0u; outputIndex < kOutputLayerSize; outputIndex++)
      {
         mOutputsStream << mOutputs[outputIndex] << " ";
      }

      mOutputsStream << "\n";
   }
}

void $Name$::captureEnd()
{
   if(mInputsStream.is_open())
   {
      mInputsStream.close();
   }
   
   if(mOutputsStream.is_open())
   {
      mOutputsStream.close();
   }
}

void $Name$::test(const char* pDataDirectory)
{
   char inputsFilePath[kFilePathMaxSize];
   snprintf(inputsFilePath, kFilePathMaxSize, "%s_inputs_", pDataDirectory);

   std::ifstream inputsFile(inputsFilePath);
   
   if(!inputsFile.is_open())
   {
      return;
   }

   char predictionFilePath[kFilePathMaxSize];
   snprintf(predictionFilePath, kFilePathMaxSize, "%s_prediction_cpp_", pDataDirectory);

   std::ofstream predictionFile(predictionFilePath);

   if(!predictionFile.is_open())
   {
      inputsFile.close();
      return;
   }

   predictionFile << std::fixed << std::showpoint << std::setprecision(kFileFloatPrecision);

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

   inputsFile.close();
   predictionFile.close();
}
