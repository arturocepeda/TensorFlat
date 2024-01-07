
import imports.nn_cpp as nn_cpp

import sys

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_generate_cpp.py <data_directory>")
  exit(1)

data_directory = sys.argv[1]

nn_cpp.generateHeaderFile(data_directory, "nn_cpp_dynamic.h")
nn_cpp.generateSourceFile(data_directory, "nn_cpp_dynamic.cpp")
