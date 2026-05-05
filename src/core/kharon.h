#ifndef KHARON_H
#define KHARON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define CK(call) do { \
  cudaError_t e_ = (call); \
  if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda error %s at %s:%d -> %s\n", #call, __FILE__, __LINE__, \
            cudaGetErrorString(e_)); \
    exit(1); \
  } \
} while (0)

#define CUBLAS_CK(call) do { \
  cublasStatus_t s_ = (call); \
  if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cublas error %s at %s:%d (%d)\n", #call, __FILE__, __LINE__, (int)s_); \
    exit(1); \
  } \
} while (0)

#define DIE(...) do { fprintf(stderr, "fatal: " __VA_ARGS__); fputc('\n', stderr); exit(1); } while (0)

#endif
