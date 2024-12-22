
#pragma once

#include <fstream>

#if !defined(TensorFlatAPI)
# define TensorFlatAPI
#endif

class TensorFlatAPI $Name$
{
private:
   static float activationLinear(float pValue);
   static float activationSigmoid(float pValue);
   static float activationReLU(float pValue);
   static float activationLeakyReLU(float pValue);

public:
   static const size_t kInputLayerSize = $InputLayerSize$;
   static const size_t kHiddenLayerSize = $HiddenLayerSize$;
   static const size_t kOutputLayerSize = $OutputLayerSize$;

   struct Inputs
   {
      enum
      {
         $InputsEnum$
      };
   };
   struct Outputs
   {
      enum
      {
         $OutputsEnum$
      };
   };

private:
   static const float kHiddenLayerWeights[kHiddenLayerSize][kInputLayerSize];
   static const float kHiddenLayerBiases[kHiddenLayerSize];

   static const float kOutputLayerWeights[kOutputLayerSize][kHiddenLayerSize];
   static const float kOutputLayerBiases[kOutputLayerSize];

   float (*mHiddenLayerActivation)(float pValue);
   float (*mOutputLayerActivation)(float pValue);

   float mInputs[kInputLayerSize];
   float mHiddenLayerValues[kHiddenLayerSize];
   float mOutputs[kOutputLayerSize];

   std::ofstream mInputsStream;
   std::ofstream mOutputsStream;

public:
   $Name$();

   inline float* getInputs()
   {
      return mInputs;
   }
   inline float* getOutputs()
   {
      return mOutputs;
   }

   void predict();

   void captureStart(const char* pDataDirectory);
   void captureSample();
   void captureEnd();

   void test(const char* pDataDirectory);
};
