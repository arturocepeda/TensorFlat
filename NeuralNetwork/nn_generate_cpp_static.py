
import imports.nn_cpp as nn_cpp

import sys

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_generate_cpp.py <data_directory>")
  exit(1)

dataDirectory = sys.argv[1]

nn_cpp.generateHeaderFile(dataDirectory, "nn_cpp_static.h")
nn_cpp.generateSourceFile(dataDirectory, "nn_cpp_static.cpp")
