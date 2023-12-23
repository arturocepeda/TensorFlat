
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

   float (*mHiddenLayerActivation)(float pValue);
   float (*mOutputLayerActivation)(float pValue);

   float mInputs[kInputLayerSize];
   float mHiddenLayerValues[kHiddenLayerSize];
   float mOutputs[kOutputLayerSize];

public:
   $Name$();

   inline float* getInputs()
   {
      return mInputs;
   }

   inline const float* getOutputs()
   {
      return mOutputs;
   }

   void predict();
};
