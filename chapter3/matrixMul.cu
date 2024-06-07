#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

float randomFloat(float min, float max) {
  float scale = rand() / (float)RAND_MAX;
  return min + scale * (max - min);
}

void randomMatrix(float *A, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      A[row * width + col] = randomFloat(-1.f, +1.f);
    }
  }
}

__global__ void matrixMulKernel(const float *A, const float *B, float *C,
                                int n) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < n && row < n) {
    float dotProduct = 0;
    for (int k = 0; k < n; k++) {
      dotProduct += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = dotProduct;
  }
}

// Assume square matrixes
void matrixMulGpu(const float *const A, const float *const B, float *C, int n) {
  float *A_d, *B_d, *C_d;

  int size = n * n * sizeof(float);

  gpuErrchk(cudaMalloc((void **)&A_d, size));
  gpuErrchk(cudaMalloc((void **)&B_d, size));
  gpuErrchk(cudaMalloc((void **)&C_d, size));

  gpuErrchk(cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice));

  // (x, y, z)
  dim3 dimGrid(ceil(n / 16.0), ceil(n / 16.0), 1);
  dim3 dimBlock(16, 16, 1);
  matrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);

  gpuErrchk(cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(A_d));
  gpuErrchk(cudaFree(B_d));
  gpuErrchk(cudaFree(C_d));
}

// Assume square matrixes
void matrixMulCpu(const float *const A, const float *const B, float *C, int n) {
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      float dotProduct = 0.f;
      for (int k = 0; k < n; k++) {
        dotProduct += A[row * n + k] * B[k * n + col];
      }
      C[row * n + col] = dotProduct;
    }
  }
}

// Can't compare floats
//
// void assertSquareMatrix(float *A, float *B, int n) {
//   for (int row = 0; row < n; row++) {
//     for (int col = 0; col < n; col++) {
//       int idx = row * n + col;
//       printf("%.2f == %.2f (%d)\n", A[idx], B[idx], A[idx] == B[idx]);
//       assert(A[idx] == B[idx]);
//     }
//   }
// }

int main(int argc, char *argv[]) {
  int n; // size of the square matrix

  if (argc != 2) {
    printf("Usage: %s <size>\n", argv[0]);
    return 1;
  }

  n = atoi(argv[1]);

  float *A = (float *)malloc(n * n * sizeof(float));
  float *B = (float *)malloc(n * n * sizeof(float));
  float *C = (float *)malloc(n * n * sizeof(float));
  float *D = (float *)malloc(n * n * sizeof(float));

  randomMatrix(A, n, n);
  randomMatrix(B, n, n);

  // matrixMulCpu(A, B, C, n);
  matrixMulGpu(A, B, D, n);

#ifdef DEBUG
  for (int row = 0; row < n; row++) {
    for (int col = 0; col < n; col++) {
      printf("%.2f ", C[row * n + col]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  return EXIT_SUCCESS;
}
