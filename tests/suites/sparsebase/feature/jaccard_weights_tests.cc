//
// Created by Taha Atahan Akyildiz on 17.04.2022.
//
#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/feature/feature_extractor.h"
#include "sparsebase/feature/jaccard_weights.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/converter/converter_order_one.h"
#include "sparsebase/utils/extractable.h"


const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
using namespace sparsebase::feature;
#include "../functionality_common.inc"


#ifndef USE_CUDA
TEST(JaccardTest, NoCuda) {
JaccardWeights<int, int, int, float> jac;
EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, true),
utils::FunctionNotFoundException);
EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, false),
utils::FunctionNotFoundException);
}
#else
TEST(JaccardTest, Jaccard) {
  JaccardWeights<int, int, int, float> jac;
  context::CUDAContext gpu_context(0);
  auto jac_array = jac.GetJaccardWeights(&global_csr, {&gpu_context}, true);
  EXPECT_EQ(jac_array->get_id(),
            format::CUDAArray<float>::get_id_static());
  converter::ConverterOrderOne<float> converter;
  auto jac_cpu_array =
      converter.Convert<format::Array<float>>(jac_array, {&cpu_context});
  EXPECT_EQ(jac_cpu_array->get_dimensions()[0], 4);
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&gpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  EXPECT_THROW(jac.GetJaccardWeights(&global_csr, {&cpu_context}, false),
               utils::FunctionNotFoundException);
}
#endif
