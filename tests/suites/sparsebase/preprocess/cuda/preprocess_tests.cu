
#include "gtest/gtest.h"

#include "sparsebase/config.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/preprocess/cuda/preprocess.cuh"
#include "sparsebase/format/format.h"
#include "sparsebase/format/cuda/format.cuh"
using namespace sparsebase;
const int n = 3;
const int nnz = 4;
int row_ptr[n+1] = {0, 2, 3, 4};
int cols[nnz] = {1,2,0,0};
int rows[nnz] = {0, 0, 1, 2};
float distribution[n] = {2.0/nnz, 1.0/nnz, 1.0/nnz};
int degrees[n] = {2, 1, 1};
format::CSR<int, int, int> global_csr(n, n, row_ptr, cols, nullptr, format::kNotOwned);
format::COO<int, int, int> global_coo(n, n, nnz, rows, cols, nullptr, format::kNotOwned);
sparsebase::context::CPUContext cpu_context;
TEST(CudaArray, CudaArrayCreat){
  EXPECT_EQ(1, 1);
}