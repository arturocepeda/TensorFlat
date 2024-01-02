
#pragma once

#include <fstream>

class $Name$
{
private:
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
   float mHiddenLayerWeights[kHiddenLayerSize][kInputLayerSize];
   float mHiddenLayerBiases[kHiddenLayerSize];

   float mOutputLayerWeights[kOutputLayerSize][kHiddenLayerSize];
   float mOutputLayerBiases[kOutputLayerSize];

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

   void loadWeightsAndBiases(const char* pInputsDirectory);
   void predict();

   void captureStart(const char* pInputsDirectory);
   void captureSample();
   void captureEnd();

   void test(const char* pInputsDirectory);
};
