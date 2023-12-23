
#include "$Name$.h"
#include <cmath>

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
}

void $Name$::predict()
{
   for(size_t i = 0u; i < kHiddenLayerSize; i++)
   {
      float sum = kHiddenLayerBiases[i];

      for(size_t j = 0; j < kInputLayerSize; j++)
      {
         sum += mInputs[j] * kHiddenLayerWeights[i][j];
      }

      mHiddenLayerValues[i] = mHiddenLayerActivation(sum);
   }

   for(size_t i = 0u; i < kOutputLayerSize; i++)
   {
      float sum = kOutputLayerBiases[i];

      for(size_t j = 0u; j < kHiddenLayerSize; j++)
      {
         sum += mHiddenLayerValues[j] * kOutputLayerWeights[i][j];
      }

      mOutputs[i] = mOutputLayerActivation(sum);
   }
}

const float $Name$::kHiddenLayerWeights[kHiddenLayerSize][kInputLayerSize] =
$HiddenLayerWeights$;
const float $Name$::kHiddenLayerBiases[kHiddenLayerSize] =
$HiddenLayerBiases$;

const float $Name$::kOutputLayerWeights[kOutputLayerSize][kHiddenLayerSize] =
$OutputLayerWeights$;
const float $Name$::kOutputLayerBiases[kOutputLayerSize] =
$OutputLayerBiases$;
