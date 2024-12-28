
#include "$Name$.h"

#include <cmath>
#include <cstdio>
#include <iomanip>


static const size_t kFilePathMaxSize = 256u;
static const int kFileFloatPrecision = 5;


float $Name$::activationLinear(float pValue)
{
   return pValue;
}
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
{
$DynamicLayerSetupInitialization$

$LayerValuesInitialization$
}

void $Name$::loadWeightsAndBiases(const char* pDataDirectory)
{
$DynamicLoadSetupCode$
}

void $Name$::predict()
{
$DynamicPredictionCode$
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
