#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/cuda.cuh"

using namespace std;
using namespace sparsebase;

void print_csr(int * row_ptr, int * col, int n){
    printf("Printing the CSR on the CPU\n");
    for (int i = 0; i < n; i++){
        printf("%d: ", i);
        for (int j = row_ptr[i]; j < row_ptr[i+1];j++){
            printf("%d ", col[j]);
        }
        printf("\n");
    }
}
__global__ void print_csr_cuda(int * row_ptr, int * col, int n){
    printf("Printing the CSR on the GPU\n");
    for (int i = 0; i < n; i++){
        printf("%d: ", i);
        for (int j = row_ptr[i]; j < row_ptr[i+1];j++){
            printf("%d ", col[j]);
        }
        printf("\n");
    }
}

int main(){

    int row_ptr[6] = {0, 2, 3, 3, 5, 6};
    int col[6] = {1, 2, 0, 0, 0, 1};
    int vals[6] = {10, 20, 30, 40, 50, 60};
    context::CUDAContext gpu_context{0};
    context::CPUContext cpu_context;

    format::CSR<int,int,int>* csr = new format::CSR<int,int,int>(5, 5, row_ptr, col, vals);

    auto converter = new utils::Converter<int,int,int>();

    auto cuda_csr = converter->ConvertConditional<format::CUDACSR<int, int, int>>(csr, &gpu_context);

    print_csr_cuda<<<1,1>>>(cuda_csr->get_row_ptr(), cuda_csr->get_col(), cuda_csr->get_dimensions()[0]);
    cudaDeviceSynchronize();

    auto cpu_csr = converter->ConvertConditional<format::CSR<int, int, int>>(cuda_csr, &cpu_context);

    print_csr(cpu_csr->get_row_ptr(), cpu_csr->get_col(), cuda_csr->get_dimensions()[0]);

    return 0;
}

