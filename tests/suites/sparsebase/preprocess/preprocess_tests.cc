#include <iostream>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <memory>

#include "gtest/gtest.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/coo.h"

#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/reorder/reorder.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/partition/partitioner.h"
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

class Degrees_DegreeDistributionTest : public ::testing::Test {
 protected:
  Degrees_DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(Degrees_DegreeDistributionTest, FeaturePreprocessTypeTests) {
  std::shared_ptr<Params1> p1(new Params1);
  std::shared_ptr<Params2> p2(new Params2);
  // Check getting feature id
  EXPECT_EQ(
      std::type_index(typeid(Degrees_DegreeDistribution<int, int, int, float>)),
      feature.get_id());
  // Getting a params object for an unset params
  EXPECT_THROW(feature.get_params(std::type_index(typeid(int))).get(),
               utils::FeatureParamsException);
  // Checking setting params of a sub-feature
  feature.set_params(Degrees<int, int, int>::get_id_static(), p1);
  EXPECT_EQ(
      feature.get_params(Degrees<int, int, int>::get_id_static()).get(),
      p1.get());
  EXPECT_NE(
      feature.get_params(Degrees<int, int, int>::get_id_static()).get(),
      p2.get());
  // Checking setting params of feature that isn't a sub-feature
  EXPECT_THROW(feature.set_params(typeid(int), p1),
               utils::FeatureParamsException);
}
class DegreesTest : public ::testing::Test {
 protected:
  Degrees<int, int, int> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(DegreesTest, AllTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetDegreesCSR implementation function
  Params1 p1;
  auto degrees_array =
      Degrees<int, int, int>::GetDegreesCSR({&global_csr}, &p1);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  // Check GetDegrees
  degrees_array = feature.GetDegrees(&global_csr, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  degrees_array = feature.GetDegrees(&global_csr, {&cpu_context}, false);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  delete[] degrees_array;
  // Check GetDegrees with conversion
  degrees_array = feature.GetDegrees(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  EXPECT_THROW(feature.GetDegrees(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check Extract
  auto feature_map = feature.Extract(&global_csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<int *>(feature_map[feature.get_id()])[i],
              degrees[i]);
  }
  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<int *>(feature_map[feature.get_id()])[i],
              degrees[i]);
  }
  EXPECT_THROW(feature.Extract(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
}
class DegreeDistributionTest : public ::testing::Test {
 protected:
  DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(DegreeDistributionTest, AllTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 1);
  EXPECT_EQ(feature.get_sub_ids()[0], std::type_index(typeid(feature)));

  // Test get_subs
  auto subs = feature.get_subs();
  // a single sub-feature
  EXPECT_EQ(subs.size(), 1);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), std::type_index(typeid(feature)));
  EXPECT_NE(subs[0], &feature);

  // Check GetDegreeDistributionCSR implementation function
  Params1 p1;
  auto distribution_array =
      DegreeDistribution<int, int, int, float>::GetDegreeDistributionCSR(
          {&global_csr}, &p1);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  //// Check GetDistribution (function matcher)
  distribution_array =
      feature.GetDistribution(&global_csr, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  distribution_array =
      feature.GetDistribution(&global_csr, {&cpu_context}, false);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  // Check GetDistribution with conversion
  distribution_array =
      feature.GetDistribution(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
  }
  delete[] distribution_array;
  EXPECT_THROW(feature.GetDistribution(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetDistribution with conversion and cached
  auto distribution_array_format =
      feature.GetDistributionCached(&global_coo, {&cpu_context}, true);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::get<1>(distribution_array_format)[i], distribution[i]);
  }
  delete[] std::get<1>(distribution_array_format);
  auto cached_data = std::get<0>(distribution_array_format);
  ASSERT_EQ(cached_data.size(), 1);
  ASSERT_EQ(cached_data[0][0]->get_id(),
            std::type_index(typeid(global_csr)));
  auto converted_csr =
      cached_data[0][0]->AsAbsolute<format::CSR<int, int, int>>();
  compare_csr(&global_csr, converted_csr);
  // Check Extract
  auto feature_map = feature.Extract(&global_csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_id()])[i],
              distribution[i]);
  }
  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_id()])[i],
              distribution[i]);
  }
}
TEST_F(Degrees_DegreeDistributionTest, Degree_DegreeDistributionTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 2);
  std::vector<std::type_index> ids = {
      Degrees<int, int, int>::get_id_static(),
      DegreeDistribution<int, int, int, float>::get_id_static()};
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(feature.get_sub_ids()[0], ids[0]);
  EXPECT_EQ(feature.get_sub_ids()[1], ids[1]);

  // Test get_subs
  auto subs = feature.get_subs();
  // two sub-feature
  EXPECT_EQ(subs.size(), 2);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), ids[0]);
  auto &feat1 = *(subs[1]);
  EXPECT_EQ(std::type_index(typeid(feat1)), ids[1]);
  EXPECT_NE(subs[0], &feature);
  EXPECT_NE(subs[1], &feature);

  // Check GetCSR implementation function
  Params1 p1;
  auto degrees_and_distribution_map =
      Degrees_DegreeDistribution<int, int, int, float>::GetCSR({&global_csr},
                                                               &p1);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]));
  auto distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]));
  auto degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  //// Check Get (function matcher)
  degrees_and_distribution_map = feature.Get(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  //// Check Get with conversion (function matcher)
  degrees_and_distribution_map = feature.Get(&global_coo, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
  EXPECT_THROW(feature.Get(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check Extract
  degrees_and_distribution_map =
      feature.Extract(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(degrees_and_distribution_map.size(), 2);
  ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
            degrees_and_distribution_map.end());
  ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
            degrees_and_distribution_map.end());
  ASSERT_NO_THROW(std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]));
  distribution_array = std::any_cast<float *>(
      degrees_and_distribution_map
          [DegreeDistribution<int, int, int, float>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]));
  degree_array = std::any_cast<int *>(
      degrees_and_distribution_map[Degrees<int, int,
                                           int>::get_id_static()]);
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(distribution_array[i], distribution[i]);
    EXPECT_EQ(degree_array[i], degrees[i]);
  }
  delete[] distribution_array;
  delete[] degree_array;
}

TEST(GraphFeatureBase, Degrees) {
  EXPECT_NO_THROW(GraphFeatureBase::GetDegrees(
      &global_csr, {&cpu_context}, true));
  auto degrees_array = GraphFeatureBase::GetDegrees(
      &global_csr, {&cpu_context}, true);
  EXPECT_EQ(std::type_index(typeid(degrees_array)),
            std::type_index(typeid(int *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
}
TEST(GraphFeatureBase, DegreesCached) {
  EXPECT_NO_THROW(GraphFeatureBase::GetDegreesCached(
      &global_csr, {&cpu_context}));
  auto output = GraphFeatureBase::GetDegreesCached(
      &global_csr, {&cpu_context});
  auto degrees_array = output.second;
  EXPECT_EQ(std::type_index(typeid(degrees_array)),
            std::type_index(typeid(int *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
  EXPECT_EQ(output.first.size(), 0);
  auto output_conv = GraphFeatureBase::GetDegreesCached(
      &global_coo, {&cpu_context});
  degrees_array = output_conv.second;
  EXPECT_EQ(output_conv.first.size(), 1);
  EXPECT_EQ(std::type_index(typeid(degrees_array)),
            std::type_index(typeid(int *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degrees_array[i], degrees[i]);
  }
}

TEST(GraphFeatureBase, DegreeDistribution) {
  EXPECT_NO_THROW(
      GraphFeatureBase::GetDegreeDistribution<float>(
          &global_csr, {&cpu_context}, true));
  auto degreeDistribution_array =
      GraphFeatureBase::GetDegreeDistribution<float>(
          &global_csr, {&cpu_context}, true);
  EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
            std::type_index(typeid(float *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
  }
}
TEST(GraphFeatureBase, DegreeDistributionCached) {
  EXPECT_NO_THROW(
      GraphFeatureBase::GetDegreeDistributionCached<
          float>(&global_csr, {&cpu_context}));
  auto output =
      GraphFeatureBase::GetDegreeDistributionCached<
          float>(&global_csr, {&cpu_context});
  auto degreeDistribution_array = output.second;
  EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
            std::type_index(typeid(float *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
  }
  EXPECT_EQ(output.first.size(), 0);
  auto output_conv =
      GraphFeatureBase::GetDegreeDistributionCached<
          float>(&global_coo, {&cpu_context});
  EXPECT_EQ(output_conv.first.size(), 1);
  degreeDistribution_array = output_conv.second;
  EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
            std::type_index(typeid(float *)));
  for (int i = 0; i < n; i++) {
    EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
  }
}
class MultiFormatKeyPreprocess : public utils::FunctionMatcherMixin<int> {
 public:
  std::tuple<std::vector<std::vector<format::Format *>>, int> GetCached(
      format::Format *f1, format::Format *f2, format::Format *f3,
      std::vector<context::Context *> contexts, bool convert_input,
      bool clear_intermediate) {
    auto p = new utils::Parameters;
    auto res = this->CachedExecute(p, std::move(contexts), convert_input,
                                   clear_intermediate, f1, f2, f3);
    return res;
  }
  MultiFormatKeyPreprocess() {
    this->RegisterFunction({format::CSR<int, int, int>::get_id_static(),
                            format::CSR<int, int, int>::get_id_static(),
                            format::CSR<int, int, int>::get_id_static()},
                           CSR_CSR_CSR);
    this->RegisterFunction({format::CSR<int, int, int>::get_id_static(),
                            format::CSC<int, int, int>::get_id_static(),
                            format::CSC<int, int, int>::get_id_static()},
                           CSR_CSC_CSC);
  }

 private:
  static int CSR_CSR_CSR(std::vector<format::Format *>, utils::Parameters *) {
    return 1;
  }
  static int CSR_CSC_CSC(std::vector<format::Format *>, utils::Parameters *) {
    return 1;
  }
};

TEST(MultiKeyFunctionMatcherMixinTest, MultiFormatKey) {
#define TYPE int, int, int
  MultiFormatKeyPreprocess x;
  auto c = std::make_shared<converter::ConverterOrderTwo<int, int, int>>();
  c->ClearConversionFunctions(
      format::CSR<int, int, int>::get_id_static(),
      format::CSC<int, int, int>::get_id_static(), false);
  context::CPUContext cpu;
  format::CSR<TYPE> *csr = global_csr.Clone()->AsAbsolute<format::CSR<TYPE>>();
  format::COO<TYPE> *coo = global_coo.Clone()->AsAbsolute<format::COO<TYPE>>();
  auto *csc = global_coo.Convert<format::CSC>();
  csr->set_converter(c);
  coo->set_converter(c);
  csc->set_converter(c);
  // No conversions needed on all three
  auto output = x.GetCached(csr, csr, csr, {&cpu}, true, false);
  auto intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for first only
  output = x.GetCached(coo, csr, csr, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 1);
  EXPECT_EQ((intermediate[0][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second only
  output = x.GetCached(csr, coo, csr, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second and third
  output = x.GetCached(csr, coo, coo, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 1);
  EXPECT_EQ((intermediate[2][0]->Is<format::CSC<TYPE>>()), true);
  // Conversion for second two-step
  output = x.GetCached(csr, csr, csc, {&cpu}, true, false);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  ASSERT_EQ(intermediate[1].size(), 2);
  EXPECT_EQ((intermediate[1][0]->Is<format::COO<TYPE>>()), true);
  EXPECT_EQ((intermediate[1][1]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  delete csc;
#undef TYPE
}

TEST(MultiKeyFunctionMatcherMixinTest, MultiFormatKeyClearIntermediate) {
#define TYPE int, int, int
  MultiFormatKeyPreprocess x;
  auto c = std::make_shared<converter::ConverterOrderTwo<int, int, int>>();
  c->ClearConversionFunctions(
      format::CSR<int, int, int>::get_id_static(),
      format::CSC<int, int, int>::get_id_static(), false);
  context::CPUContext cpu;
  format::CSR<TYPE> *csr = global_csr.Clone()->AsAbsolute<format::CSR<TYPE>>();
  format::COO<TYPE> *coo = global_coo.Clone()->AsAbsolute<format::COO<TYPE>>();
  auto *csc = global_coo.Convert<format::CSC>();
  csr->set_converter(c);
  coo->set_converter(c);
  csc->set_converter(c);
  // No conversions needed on all three
  auto output = x.GetCached(csr, csr, csr, {&cpu}, true, true);
  auto intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for first only
  output = x.GetCached(coo, csr, csr, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(std::get<1>(output), 1);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 1);
  EXPECT_EQ((intermediate[0][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[1].size(), 0);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second only
  output = x.GetCached(csr, coo, csr, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSR<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);
  // Conversion for second and third
  output = x.GetCached(csr, coo, coo, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  EXPECT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 1);
  EXPECT_EQ((intermediate[2][0]->Is<format::CSC<TYPE>>()), true);
  // Conversion for second two-step
  output = x.GetCached(csr, csr, csc, {&cpu}, true, true);
  intermediate = std::get<0>(output);
  EXPECT_EQ(intermediate.size(), 3);
  EXPECT_EQ(intermediate[0].size(), 0);
  ASSERT_EQ(intermediate[1].size(), 1);
  EXPECT_EQ((intermediate[1][0]->Is<format::CSC<TYPE>>()), true);
  EXPECT_EQ(intermediate[2].size(), 0);

  delete csc;
#undef TYPE
}
