
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
$LayerSizeDeclarations$

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
$LayerSetupDeclarations$

$LayerMemberDeclarations$

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
