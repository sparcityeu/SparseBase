#include <iostream>

#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_converter.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/cuda/format.cuh"

using namespace std;
using namespace sparsebase;

template <typename T>
void print_array(T * vals, int n){
    printf("Printing the Array on the CPU\n");
    for (int i = 0; i < n; i++){
        printf("%f ", vals[i]);
    }
    printf("\n");
}
template <typename T>
__global__ void print_array_cuda(T * vals, int n){
    printf("Printing the Array on the GPU\n");
    for (int i = 0; i < n; i++){
        printf("%f ", vals[i]);
    }
    printf("\n");
}
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

    int row_ptr[6] = {0, 2, 4, 6, 6, 6};
    int col[6] = {1, 3, 0, 2, 0, 1};
    int vals[6] = {10, 20, 30, 40, 50, 60};
    context::CUDAContext gpu_context{0};
    context::CPUContext cpu_context;

    format::CSR<int,int,int>* csr = new format::CSR<int,int,int>(5, 5, row_ptr, col, vals);

    auto graph_converter = new utils::OrderTwoConverter<int,int,int>();
    auto array_converter = new utils::OrderOneConverter<float>();


    preprocess::JaccardWeights<int, int, int, float> jac;
    auto array = jac.GetJaccardWeights({csr}, {&gpu_context, &cpu_context});

    if (array->get_context_type() == context::CPUContext::get_context_type()){
        auto cpu_array = array_converter->Convert<format::Array<float>>(array, &cpu_context);
        print_array(cpu_array->get_vals(), cpu_array->get_num_nnz());
    }
    if (array->get_context_type() == context::CUDAContext::get_context_type()){
        auto gpu_array = array_converter->Convert<format::CUDAArray<float>>(array, &gpu_context);
        print_array_cuda<<<1,1>>>(gpu_array->get_vals(), gpu_array->get_num_nnz());
    }


    auto cuda_csr = graph_converter->Convert<format::CUDACSR<int, int, int>>(csr, &gpu_context);

    print_csr_cuda<<<1,1>>>(cuda_csr->get_row_ptr(), cuda_csr->get_col(), cuda_csr->get_dimensions()[0]);
    cudaDeviceSynchronize();

    auto cpu_csr = graph_converter->Convert<format::CSR<int, int, int>>(cuda_csr, &cpu_context);

    print_csr(cpu_csr->get_row_ptr(), cpu_csr->get_col(), cuda_csr->get_dimensions()[0]);

    return 0;
}

