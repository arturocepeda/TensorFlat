
import sys
import struct

def readUInt8Value(file):
  return ord(file.read(1))

def readHighEndianUInt32Value(file):
  return struct.unpack(">I", file.read(4))[0]

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print("Usage: python convert_images_file.py <input_file>")
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

    if byte != 3:
      print("Invalid data file: dimensions should be 3 (samples, rows, columns)")
      exit(1)
      
    valuesCount = readHighEndianUInt32Value(file)
    rowsCount = readHighEndianUInt32Value(file)
    columnsCount = readHighEndianUInt32Value(file)

    with open("_inputs_", "w") as inputsFile:
      for i in range(valuesCount):
        for row in range(rowsCount):
          for column in range(columnsCount):
            pixelValue = readUInt8Value(file) / 255.0
            inputsFile.write(str(pixelValue) + " ")

        inputsFile.write("\n")
