
import imports.nn_cpp as nn_cpp

import sys

# Argument check
if len(sys.argv) < 2:
  print("Usage: nn_generate_cpp.py <name>")
  exit(1)

name = sys.argv[1]

nn_cpp.generateHeaderFile(name, "nn_cpp_static.h")
nn_cpp.generateSourceFile(name, "nn_cpp_static.cpp")
