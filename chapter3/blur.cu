#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "./lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image_write.h"

#ifndef CHANNEL_NUM
#error "CHANNEL_NUM must be defined at 1 or 3"
#endif

// #define CHANNEL_NUM 1 // Gray
// #define CHANNEL_NUM 3 // RGB

// #define BLUR_SIZE 1 // 3x3
#define BLUR_SIZE 3 // 7x7

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__global__ void blurKernel(unsigned char *out, unsigned char *in, int width,
                           int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int offset = (row * width + col) * CHANNEL_NUM;

    for (int curChannel = 0; curChannel < CHANNEL_NUM + 1; ++curChannel) {
      int pixVal = 0;
      int pixels = 0;

      for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow)
        for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
          int curRow = row + blurRow;
          int curCol = col + blurCol;

          if (curRow >= 0 && curRow < height && curCol >= 0 && curRow < width) {
            pixVal += in[(curRow * width + curCol) * CHANNEL_NUM + curChannel];
            ++pixels;
          }
        }

      out[offset + curChannel] = (unsigned char)(pixVal / pixels);
    }
  }
}

void blur(uint8_t *blurImage, uint8_t *rgbImage, int height, int width) {
  int size = width * height * CHANNEL_NUM * sizeof(uint8_t);

  unsigned char *blurImageDevice, *rgbImageDevice;

  gpuErrchk(cudaMalloc((void **)&blurImageDevice, size));
  gpuErrchk(cudaMalloc((void **)&rgbImageDevice, size));

  gpuErrchk(cudaMemcpy(rgbImageDevice, rgbImage, size, cudaMemcpyHostToDevice));

  dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
  dim3 dimBlock(16, 16, 1);
  blurKernel<<<dimGrid, dimBlock>>>(blurImageDevice, rgbImageDevice, width,
                                    height);

  gpuErrchk(
      cudaMemcpy(blurImage, blurImageDevice, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(blurImageDevice));
  gpuErrchk(cudaFree(rgbImageDevice));
}

int main(int argc, char *argv[]) {
  int width, height, bpp;

#if CHANNEL_NUM == 1
  uint8_t *rgbImage =
      stbi_load("./resources/lenna_gray.png", &width, &height, &bpp, CHANNEL_NUM);
#elif CHANNEL_NUM == 3
  uint8_t *rgbImage =
      stbi_load("./resources/lenna.png", &width, &height, &bpp, CHANNEL_NUM);
#endif

  uint8_t *blurImage =
      (uint8_t *)malloc(width * height * CHANNEL_NUM * sizeof(uint8_t));

  blur(blurImage, rgbImage, width, height);

  stbi_write_png("lenna_blur.png", width, height, CHANNEL_NUM, blurImage,
                 width * CHANNEL_NUM);

  stbi_image_free(rgbImage);
  stbi_image_free(blurImage);

  return 0;
}
