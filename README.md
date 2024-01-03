# TensorFlat
### Lightweight machine learning models for C++

Use TensorFlow to create and train your machine learning models, and then get minimal C++ code generated to benefit from the trained models in your project, without the need of adding any external dependencies to it.


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

TensorFlat allows you to create neural networks with one hidden layer, train them and generate a C++ class out of the trained model, ready to be used for predictions. For each network, both the description and the training configuration need to be specified in a JSON file called `nn.json`, which looks like this:

```json
{
  "Description": {
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
    "HiddenLayerSize": 16,
    "HiddenLayerActivation": "LeakyReLU",
    "Outputs": [
      "OutputValue01",
      "OutputValue02"
    ],
    "OutputLayerActivation": "LeakyReLU",
    "LeakyReLUAlpha": 0.01
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

* `Inputs`. The list of input parameters.
* `HiddenLayerSize`. The number of neurons in the hidden layer.
* `HiddenLayerActivation`. The activation function used in the hidden layer.
* `Outputs`. The list of output values.
* `OutputLayerActivation`. The activation function used in the output layer.
* `LeakyReLUAlpha`. The alpha value to use, in case the LeakyReLU activation function is used by one of the layers.

NOTE: the supported values for the activation function are "Sigmoid", "ReLU" and "LeakyReLU".

#### Training

* `TestSetRatio`. The proportion of the training data set that will be used for testing. 0.3 would mean 70% for training and 30% for testing.
* `LearningRate`. The learning rate to use by the gradient descent method that will train the network.
* `Epochs`. The number of iterations, or epochs, that will be executed during the training process.


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

