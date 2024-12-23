## TensorFlat Examples
### Handwritten Digits

Neural network that recognizes handwritten digits from 28x28 grayscale images, based on the MNIST data set:

https://yann.lecun.com/exdb/mnist/

To generate the required `_inputs_` and `_outputs_` files for training the network, please follow these steps:

1. Decompress the `mnist_data_set.zip` file inside the `training_data` subdirectory.

2. Run the `convert_images_file.py` script to get the inputs:  `python convert_images_file.py train-images.idx3-ubyte`.

3. Run the `convert_labels_file.py` script to get the outputs:  `python convert_labels_file.py train-labels.idx1-ubyte`.

4. Move both the generated `_inputs_` and `_outputs_` files one step below, into the `HandwrittenDigits` directory where the configuration file (`nn.json`) is stored.

Once the data set is available in the format TensorFlat expects, the network can be trained by running the `nn_train.py` script.

After having trained the network and generated the C++ class, you can use the `Test.cpp` source file for testing it, which can be run on the command line and expects the path of a compatible bitmap file as an argument. Feel free to use images from MNIST's test data set for that:

https://github.com/3omar-mostafa/MNIST-dataset-extractor/releases/tag/dataset
  
