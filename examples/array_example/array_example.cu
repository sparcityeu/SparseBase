#include <iostream>

#include "sparsebase/converter/converter.h"
#include "sparsebase/format/array.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/cuda_array_cuda.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"

using namespace std;
using namespace sparsebase;

void print_array(int *vals, int n) {
  printf("Printing the Array on the CPU\n");
  for (int i = 0; i < n; i++) {
    printf("%d ", vals[i]);
  }
  printf("\n");
}
__global__ void print_array_cuda(int *vals, int n) {
  printf("Printing the Array on the GPU\n");
  for (int i = 0; i < n; i++) {
    printf("%d ", vals[i]);
  }
  printf("\n");
}

int main() {
  int vals[6] = {10, 20, 30, 40, 50, 60};
  context::CUDAContext gpu_context{0};
  context::CPUContext cpu_context;

  format::Array<int> *array = new format::Array<int>(6, vals);

  auto cuda_array = array->Convert<format::CUDAArray>(&gpu_context);

  print_array_cuda<<<1, 1>>>(cuda_array->get_vals(),
                             cuda_array->get_dimensions()[0]);
  cudaDeviceSynchronize();

  auto cpu_array = cuda_array->Convert<format::Array>(&cpu_context);

  print_array(cpu_array->get_vals(), cuda_array->get_dimensions()[0]);

  return 0;
}
