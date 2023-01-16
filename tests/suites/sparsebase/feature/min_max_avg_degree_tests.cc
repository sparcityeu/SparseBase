#include <iostream>
#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/feature/min_degree.h"
#include "sparsebase/feature/max_degree.h"
#include "sparsebase/feature/avg_degree.h"
#include "sparsebase/feature/min_max_avg_degree.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"

using namespace sparsebase;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::feature;
#include "../functionality_common.inc"

class MinMaxAvgDegreeTest : public ::testing::Test {
 protected:
  MinMaxAvgDegree<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(MinMaxAvgDegreeTest, FeaturePreprocessTypeTests) {
  std::shared_ptr<Params1> p1(new Params1);
  std::shared_ptr<Params2> p2(new Params2);
  // Check getting feature id
  EXPECT_EQ(
      std::type_index(typeid(MinMaxAvgDegree<int, int, int, float>)),
      feature.get_id());
  // Getting a params object for an unset params
  EXPECT_THROW(feature.get_params(std::type_index(typeid(int))).get(),
               utils::FeatureParamsException);
  // Checking setting params of a sub-feature
  feature.set_params(feature::MinDegree<int, int, int>::get_id_static(), p1);
  EXPECT_EQ(feature.get_params(feature::MinDegree<int, int, int>::get_id_static())
                .get(),
            p1.get());
  EXPECT_NE(feature.get_params(feature::MinDegree<int, int, int>::get_id_static())
                .get(),
            p2.get());
  // Checking setting params of feature that isn't a sub-feature
  EXPECT_THROW(feature.set_params(typeid(int), p1),
               utils::FeatureParamsException);
}
TEST_F(MinMaxAvgDegreeTest, MinMaxAvgDegreeTests) {
  // test get_sub_ids
  EXPECT_EQ(feature.get_sub_ids().size(), 3);
  std::vector<std::type_index> ids = {
      feature::MinDegree<int, int, int>::get_id_static(),
      feature::MaxDegree<int, int, int>::get_id_static(),
      feature::AvgDegree<int, int, int, float>::get_id_static()};
  std::sort(ids.begin(), ids.end());
  EXPECT_EQ(feature.get_sub_ids()[0], ids[0]);
  EXPECT_EQ(feature.get_sub_ids()[1], ids[1]);
  EXPECT_EQ(feature.get_sub_ids()[2], ids[2]);

  // Test get_subs
  auto subs = feature.get_subs();
  // three sub-feature
  EXPECT_EQ(subs.size(), 3);
  // same type as feature but different address
  auto &feat = *(subs[0]);
  EXPECT_EQ(std::type_index(typeid(feat)), ids[0]);
  auto &feat1 = *(subs[1]);
  EXPECT_EQ(std::type_index(typeid(feat1)), ids[1]);
  auto &feat2 = *(subs[2]);
  EXPECT_EQ(std::type_index(typeid(feat2)), ids[2]);
  EXPECT_NE(subs[0], &feature);
  EXPECT_NE(subs[1], &feature);
  EXPECT_NE(subs[2], &feature);

  // Check GetCSR implementation function
  Params1 p1;
  auto minMaxAvgDegreeMap =
      MinMaxAvgDegree<int, int, int, float>::GetCSR({&global_csr},
                                                               &p1);
  ASSERT_EQ(minMaxAvgDegreeMap.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(minMaxAvgDegreeMap.find(ids[i]), minMaxAvgDegreeMap.end());
  }
  ASSERT_NO_THROW(std::any_cast<float *>(
      minMaxAvgDegreeMap[feature::AvgDegree<
          int, int, int, float>::get_id_static()]));
  auto avg_degree = std::any_cast<float *>(
      minMaxAvgDegreeMap
          [feature::AvgDegree<int, int, int, float>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]));
  auto min_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]));
  auto max_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]);

  int min_in_degrees = degrees[0];
  int max_in_degrees = degrees[0];
  for (int i = 1; i < n; ++i) {
    min_in_degrees = std::min(min_in_degrees, degrees[i]);
    max_in_degrees = std::max(max_in_degrees, degrees[i]);
  }
  float avg_in_degrees = (row_ptr[n] - row_ptr[0]) / (float) n;

  EXPECT_EQ(*min_degree, min_in_degrees);
  EXPECT_EQ(*max_degree, max_in_degrees);
  EXPECT_EQ(*avg_degree, avg_in_degrees);

  delete avg_degree;
  delete min_degree;
  delete max_degree;
  //// Check Get (function matcher)
  minMaxAvgDegreeMap = feature.Get(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(minMaxAvgDegreeMap.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(minMaxAvgDegreeMap.find(ids[i]), minMaxAvgDegreeMap.end());
  }
  ASSERT_NO_THROW(std::any_cast<float *>(
      minMaxAvgDegreeMap[feature::AvgDegree<
          int, int, int, float>::get_id_static()]));
  avg_degree = std::any_cast<float *>(
      minMaxAvgDegreeMap
          [feature::AvgDegree<int, int, int, float>::get_id_static()]);

  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]));
  min_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]);

  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]));
  max_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]);

  EXPECT_EQ(*min_degree, min_in_degrees);
  EXPECT_EQ(*max_degree, max_in_degrees);
  EXPECT_EQ(*avg_degree, avg_in_degrees);

  delete avg_degree;
  delete min_degree;
  delete max_degree;
  //// Check Get with conversion (function matcher)
  minMaxAvgDegreeMap = feature.Get(&global_coo, {&cpu_context}, true);
  ASSERT_EQ(minMaxAvgDegreeMap.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(minMaxAvgDegreeMap.find(ids[0]), minMaxAvgDegreeMap.end());
  }
  ASSERT_NO_THROW(std::any_cast<float *>(
      minMaxAvgDegreeMap[feature::AvgDegree<
          int, int, int, float>::get_id_static()]));
  avg_degree = std::any_cast<float *>(
      minMaxAvgDegreeMap
          [feature::AvgDegree<int, int, int, float>::get_id_static()]);

  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]));
  min_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]);
  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]));
  max_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]);

  EXPECT_EQ(*min_degree, min_in_degrees);
  EXPECT_EQ(*max_degree, max_in_degrees);
  EXPECT_EQ(*avg_degree, avg_in_degrees);

  delete avg_degree;
  delete min_degree;
  delete max_degree;

  EXPECT_THROW(feature.Get(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check Extract
  minMaxAvgDegreeMap =
      feature.Extract(&global_csr, {&cpu_context}, true);
  ASSERT_EQ(minMaxAvgDegreeMap.size(), 3);
  for (int i = 0; i < 3; ++i) {
    ASSERT_NE(minMaxAvgDegreeMap.find(ids[i]), minMaxAvgDegreeMap.end());
  }

  ASSERT_NO_THROW(std::any_cast<float *>(
      minMaxAvgDegreeMap[feature::AvgDegree<
          int, int, int, float>::get_id_static()]));
  avg_degree = std::any_cast<float *>(
      minMaxAvgDegreeMap
          [feature::AvgDegree<int, int, int, float>::get_id_static()]);

  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]));
  min_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MinDegree<int, int,
                                                    int>::get_id_static()]);

  ASSERT_NO_THROW(std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]));
  max_degree = std::any_cast<int *>(
      minMaxAvgDegreeMap[feature::MaxDegree<int, int,
                                            int>::get_id_static()]);

  EXPECT_EQ(*min_degree, min_in_degrees);
  EXPECT_EQ(*max_degree, max_in_degrees);
  EXPECT_EQ(*avg_degree, avg_in_degrees);

  delete avg_degree;
  delete min_degree;
  delete max_degree;
}
