#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "./lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./lib/stb_image_write.h"

#define CHANNEL_NUM 3

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

__global__ void rgbToGrayKernel(unsigned char *Pout, unsigned char *Pin,
                                int width, int height) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < height) {
    int grayOffset = row * width + col;

    int rgbOffset = grayOffset * CHANNEL_NUM;

    unsigned char r = Pin[rgbOffset];
    unsigned char g = Pin[rgbOffset + 1];
    unsigned char b = Pin[rgbOffset + 2];

    Pout[grayOffset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
  }
}

void rgbToGrayGpu(uint8_t *grayImage, uint8_t *rgbImage, int height,
                  int width) {
  int size = width * height * sizeof(uint8_t);

  unsigned char *grayImageDevice, *rgbImageDevice;

  gpuErrchk(cudaMalloc((void **)&grayImageDevice, size));
  gpuErrchk(cudaMalloc((void **)&rgbImageDevice, size * CHANNEL_NUM));

  gpuErrchk(cudaMemcpy(rgbImageDevice, rgbImage, size * CHANNEL_NUM,
                       cudaMemcpyHostToDevice));

  // (x, y, z)
  dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
  dim3 dimBlock(16, 16, 1);
  rgbToGrayKernel<<<dimGrid, dimBlock>>>(grayImageDevice, rgbImageDevice, width,
                                         height);

  gpuErrchk(
      cudaMemcpy(grayImage, grayImageDevice, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(grayImageDevice));
  gpuErrchk(cudaFree(rgbImageDevice));
}

void rgbToGrayCpu(uint8_t *grayImage, uint8_t *rgbImage, int height,
                  int width) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int grayOffset = row * width + col;

      int rgbOffset = grayOffset * CHANNEL_NUM;

      unsigned char r = rgbImage[rgbOffset];
      unsigned char g = rgbImage[rgbOffset + 1];
      unsigned char b = rgbImage[rgbOffset + 2];

      grayImage[grayOffset] = (uint8_t)(0.21f * r + 0.71f * g + 0.07f * b);
    }
  }
}

int main(int argc, char *argv[]) {
  int width, height, bpp;

  uint8_t *rgbImage =
      stbi_load("./resources/lenna.png", &width, &height, &bpp, CHANNEL_NUM);

  uint8_t *grayImage = (uint8_t *)malloc(width * height * sizeof(uint8_t));

#ifdef GPU
  printf("Running on GPU..\n");
  rgbToGrayGpu(grayImage, rgbImage, width, height);
#else
  printf("Running on CPU..\n");
  rgbToGrayCpu(grayImage, rgbImage, width, height);
#endif

#ifdef DEBUG
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++)
      printf("%3d", grayImage[row * width + col]);
    printf("\n");
  }
  printf("\n");
#endif

  stbi_write_png("lenna_gray.png", width, height, 1, grayImage, width);

  stbi_image_free(rgbImage);
  stbi_image_free(grayImage);

  return 0;
}
