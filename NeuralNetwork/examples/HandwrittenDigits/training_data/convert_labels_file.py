
import sys
import struct

def readUInt8Value(file):
  return ord(file.read(1))

def readHighEndianUInt32Value(file):
  return struct.unpack(">I", file.read(4))[0]

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python convert_labels_file.py <input_file>")
    exit(1)

  with open(sys.argv[1], "rb") as file:
    # Magic byte #1
    byte = readUInt8Value(file)

    if byte != 0:
      print("Invalid data file: 1st magic byte should be 0")
      exit(1)

    # Magic byte #2
    byte = readUInt8Value(file)

    if byte != 0:
      print("Invalid data file: 2st magic byte should be 0")
      exit(1)

    # Data format
    byte = readUInt8Value(file)

    if byte != 8:
      print("Invalid data file: data format should be 8 (unsigned byte)")
      exit(1)

    # Dimensions
    byte = readUInt8Value(file)

    if byte != 1:
      print("Invalid data file: dimensions should be 1 (samples)")
      exit(1)
      
    valuesCount = readHighEndianUInt32Value(file)

    with open("_outputs_", "w") as outputsFile:
      for i in range(valuesCount):
        byte = readUInt8Value(file)

        for digit in range(10):
          if digit == byte:
            outputsFile.write("1 ")
          else:
            outputsFile.write("0 ")

        outputsFile.write("\n")
