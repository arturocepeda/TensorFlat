
class $Name$
{
private:
   static float activationSigmoid(float pValue)
   {
      return 1.0f / (1.0f + expf(-pValue));
   }
   static float activationReLU(float pValue)
   {
      return pValue > 0.0f ? pValue : 0.0f;
   }
   static float activationLeakyReLU(float pValue)
   {
      static const float kAlpha = 0.01f;
      return pValue > 0.0f ? pValue : (pValue * kAlpha);
   }

public:
   static const size_t kInputLayerSize = $InputLayerSize$;
   static const size_t kHiddenLayerSize = $HiddenLayerSize$;
   static const size_t kOutputLayerSize = $OutputLayerSize$;

   enum Inputs
   {
      $InputsEnum$
   };
   enum Outputs
   {
      $OutputsEnum$
   };

private:
   static const float kHiddenLayerWeights[kHiddenLayerSize][kInputLayerSize];
   static const float kHiddenLayerBiases[kHiddenLayerSize];

   static const float kOutputLayerWeights[kOutputLayerSize][kHiddenLayerSize];
   static const float kOutputLayerBiases[kOutputLayerSize];

   float (*mActivation)(float pValue);

   float mInputs[kInputLayerSize];
   float mHiddenLayerValues[kHiddenLayerSize];
   float mOutputs[kOutputLayerSize];

public:
   $Name$()
      : mActivation(activationLeakyReLU)
   {
   }

   float* getInputs()
   {
      return mInputs;
   }

   const float* getOutputs()
   {
      return mOutputs;
   }

   void predict()
   {
      for(size_t i = 0u; i < kHiddenLayerSize; i++)
      {
         float sum = kHiddenLayerBiases[i];

         for(size_t j = 0; j < kInputLayerSize; j++)
         {
            sum += mInputs[j] * kHiddenLayerWeights[i][j];
         }

         mHiddenLayerValues[i] = mActivation(sum);
      }

      for(size_t i = 0u; i < kOutputLayerSize; i++)
      {
         float sum = kOutputLayerBiases[i];

         for(size_t j = 0u; j < kHiddenLayerSize; j++)
         {
            sum += mHiddenLayerValues[j] * kOutputLayerWeights[i][j];
         }

         mOutputs[i] = mActivation(sum);
      }
   }
};
