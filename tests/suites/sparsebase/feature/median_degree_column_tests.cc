#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/median_degree_column.h"
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

class MedianDegreeColumnTest : public ::testing::Test {
 protected:
  MedianDegreeColumn<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(MedianDegreeColumnTest, AllTests) {
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

  // Check GetMedianDegreeColumnCSC implementation function
  Params1 p1;
  auto median_degree =
      MedianDegreeColumn<int, int, int, float>::GetMedianDegreeColumnCSC(
          {&global_csc}, &p1);

  std::vector<int> test_degrees;
  test_degrees.reserve(n);
  for (int i = 0; i < n; i++) {
    int current_degree = cols[i + 1] - cols[i];
    test_degrees.push_back(current_degree);
  }
  std::sort(test_degrees.begin(), test_degrees.end());

  auto median_in_degrees = 0.0;
  if (n % 2 == 0) {
    median_in_degrees = (test_degrees[n / 2 - 1] + test_degrees[n / 2]) / 2.0;
  }
  else {
    median_in_degrees = test_degrees[n / 2];
  }
/*
  //EXPECT_NEAR(*median_degree, median_in_degrees, 0.000001);aaaa

  delete median_degree;
  //// Check GetMedianDegree (function matcher)
  median_degree =
      feature.GetMedianDegreeColumn(&global_csc, {&cpu_context}, true);
  EXPECT_NEAR(*median_degree, median_in_degrees, 0.000001);
  delete median_degree;

  median_degree =
      feature.GetMedianDegreeColumn(&global_csc, {&cpu_context}, false);
  EXPECT_NEAR(*median_degree, median_in_degrees, 0.000001);
  delete median_degree;

  // Check GetMedianDegree with conversion
  median_degree =
      feature.GetMedianDegreeColumn(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*median_degree, median_in_degrees, 0.000001);
  delete median_degree;
  EXPECT_THROW(feature.GetMedianDegreeColumn(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetMedianDegree with conversion and cached
  auto median_degree_format =
      feature.GetMedianDegreeColumnCached(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*std::get<1>(median_degree_format), median_in_degrees, 0.000001);
  delete std::get<1>(median_degree_format);

  auto cached_data = std::get<0>(median_degree_format);
  ASSERT_EQ(cached_data.size(), 1);
  ASSERT_EQ(cached_data[0][0]->get_id(), std::type_index(typeid(global_csc)));
  auto converted_csc =
      cached_data[0][0]->AsAbsolute<format::CSC<int, int, int>>();
  compare_csc(&global_csc, converted_csc);
  // Check Extract
  auto feature_map = feature.Extract(&global_csc, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
              median_in_degrees, 0.000001);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
              median_in_degrees, 0.000001);
              */
}
