
#include "gtest/gtest.h"

#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/context/cuda/context.cuh"
#include "sparsebase/format/cuda/format.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/converter/converter.h"
using namespace sparsebase;
const int n = 3;
const int nnz = 4;
int row_ptr[n + 1] = {0, 2, 3, 4};
int cols[nnz] = {1, 2, 0, 0};
int rows[nnz] = {0, 0, 1, 2};
float distribution[n] = {2.0 / nnz, 1.0 / nnz, 1.0 / nnz};
int degrees[n] = {2, 1, 1};
format::CSR<int, int, int> global_csr(n, n, row_ptr, cols, nullptr,
                                      format::kNotOwned);
format::COO<int, int, int> global_coo(n, n, nnz, rows, cols, nullptr,
                                      format::kNotOwned);
sparsebase::context::CPUContext cpu_context;
TEST(JaccardTest, Jaccard) {
  sparsebase::preprocess::JaccardWeights<int, int, int, float> jac;
  context::cuda::CUDAContext gpu_context(0);
  auto jac_array = jac.GetJaccardWeights(&global_csr, {&gpu_context}, true);
  EXPECT_EQ(jac_array->get_format_id(),
            format::cuda::CUDAArray<float>::get_format_id_static());
  utils::converter::ConverterOrderOne<float> converter;
  auto jac_cpu_array =
      converter.Convert<format::Array<float>>(jac_array, {&gpu_context});
  EXPECT_EQ(jac_cpu_array->get_dimensions()[0], 4);
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
}