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

// Each thread handles 1 element
__global__ void vecAddKernel(const float *A, const float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// Each thread handles 2 consecutives element
__global__ void vecAddKernel2(const float *A, const float *B, float *C, int n) {
  int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;

  if (i < n) {
    C[i] = A[i] + B[i];
  }

  i += 1;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

// Each thread handles 2 elements, the elements are in consecutive blocks.
__global__ void vecAddKernel3(const float *A, const float *B, float *C, int n) {
  int idx = (blockDim.x * blockIdx.x * 2) + threadIdx.x;

  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }

  idx += blockDim.x;

  if (idx < n) {
    C[idx] = A[idx] + B[idx];
  }
}

void vecAddGpu(float *A_h, float *B_h, float *C_h, int n) {
  int size = n * sizeof(float);

  float *A_d, *B_d, *C_d;
  gpuErrchk(cudaMalloc((void **)&A_d, size));
  gpuErrchk(cudaMalloc((void **)&B_d, size));
  gpuErrchk(cudaMalloc((void **)&C_d, size));

  gpuErrchk(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(C_d, C_h, size, cudaMemcpyHostToDevice));

  // <<<block,threads>>>  (thead multiple 32 for efficiency)
  vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

  gpuErrchk(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(A_d));
  gpuErrchk(cudaFree(B_d));
  gpuErrchk(cudaFree(C_d));
}

void vecAddCpu(float *A_h, float *B_h, float *C_h, int n) {
  for (int i = 0; i < n; i++) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

float floatRand(float min, float max) {
  float scale = rand() / (float)RAND_MAX;
  return min + scale * (max - min);
}

void initVec(float *v, int n) {
  for (int i = 0; i < n; i++) {
    v[i] = floatRand(0, 100);
  }
}

void printVec(float *v, int n) {
  if (n > 0) {
    printf("[%.2f", v[0]);
    for (int i = 1; i < n; i++) {
      printf(", %.2f", v[i]);
    }
    printf("]\n");
  } else {
    printf("[]\n");
  }
}

void assertEqVec(float *A, float *B, int n) {
  for (int i = 0; i < n; i++) {
    assert(A[i] == B[i]);
  }
}

int main() {
  const int n = 1000;

  float *A_h = (float *)malloc(n * sizeof(float));
  float *B_h = (float *)malloc(n * sizeof(float));
  float *C_h = (float *)malloc(n * sizeof(float));
  float *D_h = (float *)malloc(n * sizeof(float));

  initVec(A_h, n);
  initVec(B_h, n);

  vecAddGpu(A_h, B_h, C_h, n);
  vecAddCpu(A_h, B_h, D_h, n);

  assertEqVec(C_h, D_h, n);
  printf("vecAddGpu is equal to vecAddCpu\n");
}
