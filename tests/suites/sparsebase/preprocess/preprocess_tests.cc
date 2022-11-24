#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/partition/partitioner.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/feature/degrees.h"
#include "sparsebase/feature/degree_distribution.h"
#include "sparsebase/utils/exception.h"
#ifdef USE_CUDA
#include "sparsebase/converter/converter_cuda.cuh"
#include "sparsebase/converter/converter_order_one_cuda.cuh"
#include "sparsebase/converter/converter_order_two_cuda.cuh"
#include "sparsebase/format/cuda_csr_cuda.cuh"
#include "sparsebase/format/cuda_array_cuda.cuh"
#endif

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";


using namespace sparsebase;
using namespace sparsebase::preprocess;
using namespace sparsebase::reorder;
using namespace sparsebase::partition;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(TypeIndexHash, Basic) {
  utils::TypeIndexVectorHash hasher;
  // Empty vector
  std::vector<std::type_index> vec;
  EXPECT_EQ(hasher(vec), 0);
  // Vector with values
  vec.push_back(typeid(int));
  vec.push_back(typeid(double));
  vec.push_back(typeid(float));
  size_t hash = 0;
  for (auto tid : vec) {
    hash += tid.hash_code();
  }
  EXPECT_EQ(hash, hasher(vec));
}



#ifndef USE_CUDA
TEST(JaccardTest, NoCuda) {
  JaccardWeights<int, int, int, float> jac;
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, true),
               utils::FunctionNotFoundException);
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, false),
               utils::FunctionNotFoundException);
}
#endif
