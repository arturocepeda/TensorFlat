
#include "HandwrittenDigits.h"

#include <iostream>
#include <vector>
#include <cassert>


#pragma pack(push, 1)
struct BMPHeader
{
   char mSignature[2];
   int32_t mFileSize;
   int32_t mReserved;
   int32_t mDataOffset;
   int32_t mHeaderSize;
   int32_t mImageWidth;
   int32_t mImageHeight;
   int16_t mPlanes;
   int16_t mBitsPerPixel;
};
#pragma pack(pop)

static_assert
(
   sizeof(BMPHeader) ==
      sizeof(char[2]) +
      sizeof(int32_t) +
      sizeof(int32_t) +
      sizeof(int32_t) +
      sizeof(int32_t) +
      sizeof(int32_t) +
      sizeof(int32_t) +
      sizeof(int16_t) +
      sizeof(int16_t),
   "No padding allowed for the BMPHeader struct"
);


int main(int argc, char* argv[])
{
   if(argc < 2)
   {
      std::cout << "Usage: Test <bmp_file>" << std::endl;
      return 1;
   }

   std::ifstream file(argv[1], std::ios::binary);

   if(!file)
   {
      std::cout << "The specified input file could not be open" << std::endl;
      return 1;
   }

   BMPHeader bmpHeader;
   file.read((char*)&bmpHeader, sizeof(bmpHeader));

   if(bmpHeader.mSignature[0] != 'B' || bmpHeader.mSignature[1] != 'M')
   {
      std::cout << "The specified input file is not a valid BMP file" << std::endl;
      file.close();
      return 1;
   }

   if(bmpHeader.mImageWidth != 28 || bmpHeader.mImageHeight != 28 || bmpHeader.mBitsPerPixel != 8)
   {
      std::cout << "The expected image file is 28x28, 8 bits per pixel" << std::endl;
      file.close();
      return 1;
   }

   const int32_t pixelDataSize = bmpHeader.mImageWidth * bmpHeader.mImageHeight;
   assert(pixelDataSize == HandwrittenDigits::kInputLayerSize);
   
   std::vector<uint8_t> pixelData;
   pixelData.resize(pixelDataSize);

   file.seekg(bmpHeader.mDataOffset, std::ios::beg);

   for(int32_t row = bmpHeader.mImageHeight - 1; row >= 0; row--)
   {
      char* pixelDataCursor = (char*)pixelData.data() + (row * bmpHeader.mImageHeight);
      file.read(pixelDataCursor, bmpHeader.mImageWidth);
   }

   file.close();

   HandwrittenDigits nn;
   float* nnInputs = nn.getInputs();

   for(size_t i = 0u; i < pixelData.size(); i++)
   {
      nnInputs[i] = pixelData[i] / 255.0f;
   }
   
   nn.predict();

   float digitHighestValue = 0.0f;
   size_t digit = 10u;

   for(size_t i = 0u; i < HandwrittenDigits::kOutputLayerSize; i++)
   {
      const float output = nn.getOutputs()[i];

      if(output > 0.5f && output > digitHighestValue)
      {
         digitHighestValue = output;
         digit = i;
      }
   }

   if(digit < 10u)
   {
      std::cout << "Digit: " << digit << std::endl;
   }
   else
   {
      std::cout << "Unrecognized digit" << std::endl;
   }
   
   return 0;
}
