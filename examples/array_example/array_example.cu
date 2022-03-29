#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/cuda/format.cuh"

using namespace std;
using namespace sparsebase;

void print_array(int * vals, int n){
    printf("Printing the Array on the CPU\n");
    for (int i = 0; i < n; i++){
        printf("%d ", vals[i]);
    }
    printf("\n");
}
__global__ void print_array_cuda(int * vals, int n){
    printf("Printing the Array on the GPU\n");
    for (int i = 0; i < n; i++){
        printf("%d ", vals[i]);
    }
    printf("\n");
}

int main(){

    int vals[6] = {10, 20, 30, 40, 50, 60};
    context::CUDAContext gpu_context{0};
    context::CPUContext cpu_context;

    format::Array<int>* array = new format::Array<int>(6, vals);

    auto converter = new utils::OrderOneConverter<int>();

    auto cuda_array = converter->Convert<format::CUDAArray<int>>(array, &gpu_context);

    print_array_cuda<<<1,1>>>(cuda_array->get_vals(), cuda_array->get_dimensions()[0]);
    cudaDeviceSynchronize();

    auto cpu_array = converter->Convert<format::Array<int>>(cuda_array, &cpu_context);

    print_array(cpu_array->get_vals(), cuda_array->get_dimensions()[0]);

    return 0;
}

