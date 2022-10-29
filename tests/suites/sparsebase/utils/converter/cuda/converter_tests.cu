#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"
using namespace sparsebase;
using namespace utils::converter;
using namespace utils::converter::cuda;
using namespace format;
using namespace format::cuda;
using namespace context;
using namespace context::cuda;

// The arrays defined here are for two matrices
// One in csr format one in coo format
// These are known to be equivalent (converted using scipy)
const int n = 12;
const int m = 9;
const int nnz = 7;
int coo_row[7]{0, 0, 1, 3, 5, 10, 11};
int coo_col[7]{0, 2, 1, 3, 3, 8, 7};
int coo_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csr_row_ptr[13]{0, 2, 3, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7};
int csr_col[7]{0, 2, 1, 3, 3, 8, 7};
int csr_vals[7]{3, 5, 7, 9, 15, 11, 13};
int csc_col_ptr[13]{0, 1, 2, 3, 5, 5, 5, 5, 6, 7, 7, 7, 7};
int csc_row[7]{0, 1, 0, 3, 5, 11, 10};
int csc_vals[7]{3, 7, 5, 9, 15, 13, 11};
#define TYPE int, int, int

class CUDAFormatsFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    csr = new CSR<int, int, int>(n, m, csr_row_ptr, csr_col, csr_vals,
                                 sparsebase::format::kNotOwned);
    array = new Array<int>(nnz, csr_vals, sparsebase::format::kNotOwned);
    cudaMalloc(&cuda_col, nnz * sizeof(int));
    cudaMemcpy(cuda_col, csr_col, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_vals, nnz * sizeof(int));
    cudaMemcpy(cuda_vals, csr_vals, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_row_ptr, (n + 1) * sizeof(int));
    cudaMemcpy(cuda_row_ptr, csr_row_ptr, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMalloc(&cuda_arr_vals, (nnz) * sizeof(int));
    cudaMemcpy(cuda_arr_vals, csr_vals, (n + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
  }
  void TearDown() override {
    delete csr;
    delete array;
    cudaFree(cuda_col);
    cudaFree(cuda_row_ptr);
    cudaFree(cuda_vals);
    cudaFree(cuda_arr_vals);
  }
  template <typename T>
  void compare_arrays_cuda_cpu(T* cuda_ptr, T* cpu_ptr, int size) {
    T* cuda_copy_on_cpu = new T[size];
    cudaMemcpy(cuda_copy_on_cpu, cuda_ptr, size * sizeof(T),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(cuda_copy_on_cpu[i], cpu_ptr[i]);
    }
    delete[] cuda_copy_on_cpu;
  }
  template <typename ID, typename NNZ, typename Value>
  void compare_cuda_cpu_csr(CUDACSR<ID, NNZ, Value>* cudacsr,
                            CSR<ID, NNZ, Value>* cpucsr) {
    compare_arrays_cuda_cpu(cudacsr->get_row_ptr(), cpucsr->get_row_ptr(),
                            cpucsr->get_dimensions()[0] + 1);
    compare_arrays_cuda_cpu(cudacsr->get_col(), cpucsr->get_col(),
                            cpucsr->get_num_nnz());
    compare_arrays_cuda_cpu(cudacsr->get_vals(), cpucsr->get_vals(),
                            cpucsr->get_num_nnz());
  }
  int* cuda_col;
  int* cuda_row_ptr;
  int* cuda_vals;
  int* cuda_arr_vals;
  CSR<TYPE>* csr;
  Array<int>* array;
  CUDAContext gpu0{0};
  CPUContext cpu;
  ConverterOrderTwo<TYPE> c2;
  ConverterOrderOne<int> c1;
};
TEST_F(CUDAFormatsFixture, CUDACSR) {
  ConversionChain chain;
  format::Format* output_format;
  // Can't convert for bad context
  chain = c2.GetConversionChain(csr->get_format_id(), csr->get_context(),
                                CUDACSR<TYPE>::get_format_id_static(), {&cpu});
  EXPECT_EQ(chain.has_value(), false);
  // Can convert
  chain = c2.GetConversionChain(csr->get_format_id(), csr->get_context(),
                                CUDACSR<TYPE>::get_format_id_static(), {&gpu0});
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()),
            CUDAContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(csr, &gpu0);
  EXPECT_EQ(output_format->get_format_id(),
            (CUDACSR<TYPE>::get_format_id_static()));
  compare_cuda_cpu_csr(output_format->AsAbsolute<CUDACSR<TYPE>>(), csr);
  // Can't convert to CSR bad context
  chain = c2.GetConversionChain(CUDACSR<TYPE>::get_format_id_static(),
                                output_format->get_context(),
                                CSR<TYPE>::get_format_id_static(), {&gpu0});
  EXPECT_EQ(chain.has_value(), false);
  // Can convert to CSR
  chain = c2.GetConversionChain(CUDACSR<TYPE>::get_format_id_static(),
                                output_format->get_context(),
                                CSR<TYPE>::get_format_id_static(), {&cpu});
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()),
            CPUContext::get_context_type());
  format::Format* output_format_csr;
  output_format_csr =
      (std::get<0>(std::get<0>(*chain)[0]))(output_format, &cpu);
  EXPECT_EQ(output_format_csr->get_format_id(),
            (CSR<TYPE>::get_format_id_static()));
  compare_cuda_cpu_csr(output_format->AsAbsolute<CUDACSR<TYPE>>(),
                       output_format_csr->AsAbsolute<CSR<TYPE>>());
  delete output_format;
  delete output_format_csr;
}

TEST_F(CUDAFormatsFixture, CUDAArray) {
  ConversionChain chain;
  format::Format* output_format;
  // Can't convert for bad context
  chain = c1.GetConversionChain(array->get_format_id(), array->get_context(),
                                CUDAArray<int>::get_format_id_static(), {&cpu});
  EXPECT_EQ(chain.has_value(), false);
  // Can convert
  chain =
      c1.GetConversionChain(array->get_format_id(), array->get_context(),
                            CUDAArray<int>::get_format_id_static(), {&gpu0});
  ASSERT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()),
            CUDAContext::get_context_type());
  output_format = (std::get<0>(std::get<0>(*chain)[0]))(array, &gpu0);
  EXPECT_EQ(output_format->get_format_id(),
            (CUDAArray<int>::get_format_id_static()));
  compare_arrays_cuda_cpu(
      output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
      array->get_vals(), array->get_num_nnz());
  // Can't convert to Array bad context
  chain = c1.GetConversionChain(CUDAArray<int>::get_format_id_static(),
                                output_format->get_context(),
                                Array<int>::get_format_id_static(), {&gpu0});
  EXPECT_EQ(chain.has_value(), false);
  // Can convert to Array
  chain = c1.GetConversionChain(CUDAArray<int>::get_format_id_static(),
                                output_format->get_context(),
                                Array<int>::get_format_id_static(), {&cpu});
  EXPECT_EQ(chain.has_value(), true);
  EXPECT_EQ((std::get<0>(*chain)).size(), 1);
  EXPECT_EQ((std::get<1>(std::get<0>(*chain)[0])->get_context_type_member()),
            CPUContext::get_context_type());
  format::Format* output_format_csr;
  output_format_csr =
      (std::get<0>(std::get<0>(*chain)[0]))(output_format, &cpu);
  EXPECT_EQ(output_format_csr->get_format_id(),
            (Array<int>::get_format_id_static()));
  compare_arrays_cuda_cpu(
      output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
      output_format_csr->AsAbsolute<Array<int>>()->get_vals(),
      output_format->get_num_nnz());
  delete output_format;
  delete output_format_csr;
}

TEST_F(CUDAFormatsFixture, CUDAArrayCached) {
  ConversionChain chain;
  std::vector<format::Format*> output_formats;
  // Can't convert for bad context
  EXPECT_THROW((c1.ConvertCached(array, CUDAArray<int>::get_format_id_static(),
                                 &cpu, false)),
               utils::ConversionException);

  output_formats = c1.ConvertCached(
      array, CUDAArray<int>::get_format_id_static(), &gpu0, false);
  EXPECT_EQ(output_formats.size(), 1);
  compare_arrays_cuda_cpu(
      output_formats[0]->AsAbsolute<CUDAArray<int>>()->get_vals(),
      array->get_vals(), array->get_num_nnz());
  delete output_formats[0];

  output_formats = c1.ConvertCached(
      array, CUDAArray<int>::get_format_id_static(), {&gpu0, &cpu}, false);
  EXPECT_EQ(output_formats.size(), 1);
  compare_arrays_cuda_cpu(
      output_formats[0]->AsAbsolute<CUDAArray<int>>()->get_vals(),
      array->get_vals(), array->get_num_nnz());
}

TEST_F(CUDAFormatsFixture, CUDAArrayMultiContext) {
  ConversionChain chain;
  format::Format* output_format;
  // Can't convert for bad context
  EXPECT_THROW(
      (c1.Convert(array, CUDAArray<int>::get_format_id_static(), &cpu, false)),
      utils::ConversionException);

  auto cuda_array = c1.Convert<CUDAArray<int>>(array, {&gpu0, &cpu}, false);
  compare_arrays_cuda_cpu(cuda_array->get_vals(), array->get_vals(),
                          array->get_num_nnz());
  delete cuda_array;

  output_format = c1.Convert(array, CUDAArray<int>::get_format_id_static(),
                             {&gpu0, &cpu}, false);
  compare_arrays_cuda_cpu(
      output_format->AsAbsolute<CUDAArray<int>>()->get_vals(),
      array->get_vals(), array->get_num_nnz());
  delete output_format;
  cuda_array = array->Convert<CUDAArray>({&gpu0, &cpu}, false);
  compare_arrays_cuda_cpu(cuda_array->get_vals(), array->get_vals(),
                          array->get_num_nnz());
  delete cuda_array;
}
