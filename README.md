# TensorFlat
### Lightweight machine learning models for C++

Use TensorFlow to create and train your machine learning models, and then get minimal C++ code generated to benefit from the trained models in your project, without the need of adding any external dependencies to it. The generated code is also platform-agnostic, and does not require any dynamic memory allocations.


## Requirements for model training
### Python

TensorFlat is based on TensorFlow, a widely used open-source library for machine learning. The first step is to get a **64-bit** version of Python installed:

https://www.python.org/downloads/

Currently, the recommended version is the **3.11**.

### TensorFlow

```console
pip install tensorflow
```

### pandas

```console
pip install pandas
```

### Matplotlib

```console
pip install matplotlib
```

### scikit-learn

```console
pip install scikit-learn
```


## Documentation
### Neural networks

TensorFlat allows you to create neural networks with any number of hidden layers, train them and generate a C++ class out of the trained model, ready to be used for predictions. For each network, both the description and the training configuration need to be specified in a JSON file called `nn.json`, which looks like this:

```json
{
  "Description": {
    "Name": "NNSample",
    "Inputs": [
      "InputParameter01",
      "InputParameter02",
      "InputParameter03",
      "InputParameter04",
      "InputParameter05",
      "InputParameter06",
      "InputParameter07",
      "InputParameter08"
    ],
    "HiddenLayers": [
      {
        "HiddenLayerSize": 16,
        "HiddenLayerActivation": "LeakyReLU"
      }
    ],
    "Outputs": [
      "OutputValue01",
      "OutputValue02"
    ],
    "OutputLayerActivation": "LeakyReLU",
    "LeakyReLUNegativeSlope": 0.01
  },
  "Training": {
    "TestSetRatio": 0.3,
    "LearningRate": 0.001,
    "Epochs": 100
  }
}
```

Let's now go through the properties, one by one, to see what each one of them defines:

#### Description

* `Name`. The name of the generated C++ class.
* `Inputs`. The list of input parameters.
* `HiddenLayers`. The list of hidden layers. For each one:
  * `HiddenLayerSize`. The number of neurons in that hidden layer.
  * `HiddenLayerActivation`. The activation function used in that hidden layer.
* `Outputs`. The list of output values.
* `OutputLayerActivation`. The activation function used in the output layer.
* `LeakyReLUNegativeSlope`. The negative slope value to use, in case the LeakyReLU activation function is used by any of the layers.

The supported values for the activation function are *"Linear"*, *"Sigmoid"*, *"ReLU"* and *"LeakyReLU"*.

#### Training

* `TestSetRatio`. The proportion of the training data set that will be used for testing. 0.3 would mean 70% for training and 30% for testing.
* `LearningRate`. The learning rate to use by the gradient descent method that will train the network.
* `Epochs`. The number of iterations, or epochs, that will be executed during the training process.

#### Generating the C++ class

There are two different C++ versions of the network class that can be generated, depending on the use case:

* Static. The weights and biases are defined in static const arrays and cannot be modified at runtime. The only possibility in this case would be to use trained networks exclusively for predictions.

* Dynamic. The weights and biases can be modified at runtime, so this version should be used if the weights and biases need to be adjusted dynamically during the execution.

For each version, there is a different script: `nn_generate_cpp_static.py` and `nn_generate_cpp_dynamic.py`, respectively. After running any of them, with the directory where the JSON file is located as an argument, a `.h` and a `.cpp` file will be generated on the specified directory:

```console
python nn_generate_cpp_static.py ./NNSample/
```

```console
python nn_generate_cpp_dynamic.py ./NNSample/
```

#### Training data collection

Both the static and the dynamic version of the C++ class provide the following methods to capture training data from the application:

```cpp
   void captureStart(const char* pDataDirectory);
   void captureSample();
   void captureEnd();
```

There are also getters to access the arrays of inputs and outputs (`getInputs` and `getOuputs`), so the corresponding values can be set before capturing each sample. The captured values end up then on the given data directory, in the files `_inputs_` and `_outputs_`, respectively.

Alternatively, the data can be gathered externally, as long as it matches the expected format (a text file with space separated values, where each line represents one sample):

```
<Sample0-Value0> <Sample0-Value1> <Sample0-Value2> ...
<Sample1-Value0> <Sample1-Value1> <Sample1-Value2> ...
...
```

#### Training the network - Static version

To train the network and generate a C++ class with the trained version, you can follow these steps:

1. Collect the training data. It can be either by capturing data directly from within the application, using the (still untrained) generated class, or externally.

2. Set the desired training parameters in the `nn.json` file.

3. Run the `nn_train.py` script:  `python nn_train.py ./NNSample/`

4. (Optional) Execute the `nn_predict.py` script to have the network generate a file with the predicted outputs (`_prediction_`), calculated out of the inputs provided in the `_inputs_` file.

5. Get the C++ class generated:  `python nn_generate_cpp_static.py ./NNSample/`

6. (Optional) Call the `test` method from your application to have the C++ network class generate a file with the predicted outputs (`_prediction_cpp_`), again calculated from the `_inputs_` file, to make sure that they match the results from the script.


#### Dynamic version

The dynamic version of the generated C++ class is mostly identical to the static version, with the only difference being that the dynamic version has the capability of loading weights and biases on the fly. There are two different ways of updating them:

1. Training the network using `nn_train.py`, and calling the following method from the application:  `void loadWeightsAndBiases(const char* pDataDirectory);`. Note that, in this case, the class expects the output files from the script to be present on the given data directory.

2. Manipulating the values from within the application, by using the getters to access the arrays.


## License

TensorFlat is distributed with a *zlib* license, and is free to use for both non-commercial and commercial projects:

```
Copyright (c) 2023-2024 Arturo Cepeda Pérez

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
```

