
#include "$Name$.h"

#include <cmath>
#include <cstring>


static const size_t kFilePathMaxSize = 256u;


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
   static const float kNegativeSlope = $LeakyReLUNegativeSlope$f;
   return pValue > 0.0f ? pValue : (pValue * kNegativeSlope);
}


$Name$::$Name$()
{
$LayerValuesInitialization$
}

void $Name$::predict()
{
$StaticPredictionCode$
}

void $Name$::captureStart(const char* pDataDirectory)
{
   char inputsFilePath[kFilePathMaxSize];
   snprintf(inputsFilePath, kFilePathMaxSize, "%s_inputs_", pDataDirectory);

   mInputsStream = std::ofstream(inputsFilePath, std::ios_base::app);

   char outputsFilePath[kFilePathMaxSize];
   snprintf(outputsFilePath, kFilePathMaxSize, "%s_outputs_", pDataDirectory);

   mOutputsStream = std::ofstream(outputsFilePath, std::ios_base::app);
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


$StaticLayerSetupDefinitions$
