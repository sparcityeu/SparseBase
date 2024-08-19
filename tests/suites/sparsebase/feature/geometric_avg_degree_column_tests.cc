#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>
#include <cmath>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/geometric_avg_degree_column.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csc.h"
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

class GeometricAvgDegreeColumnTest : public ::testing::Test {
 protected:
  GeometricAvgDegreeColumn<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(GeometricAvgDegreeColumnTest, AllTests) {
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

  // Check GetGeometricAvgDegreeColumncsc implementation function
  Params1 p1;
  auto geometric_avg_degree =
      GeometricAvgDegreeColumn<int, int, int, float>::GetGeometricAvgDegreeColumnCSC(
          {&global_csc}, &p1);

  auto geometric_avg_in_degrees = 0.0;
  for (int i = 0; i < n; i++) {
    geometric_avg_in_degrees += log(col_ptr[i + 1] - col_ptr[i]);
  }
  geometric_avg_in_degrees = exp(geometric_avg_in_degrees / n);

  EXPECT_NEAR(*geometric_avg_degree, geometric_avg_in_degrees, 0.000001);

  delete geometric_avg_degree;
  //// Check GetGeometricAvgDegreeColumn (function matcher)
  geometric_avg_degree =
      feature.GetGeometricAvgDegreeColumn(&global_csc, {&cpu_context}, true);
  EXPECT_NEAR(*geometric_avg_degree, geometric_avg_in_degrees, 0.000001);
  delete geometric_avg_degree;

  geometric_avg_degree =
      feature.GetGeometricAvgDegreeColumn(&global_csc, {&cpu_context}, false);
  EXPECT_NEAR(*geometric_avg_degree, geometric_avg_in_degrees, 0.000001);
  delete geometric_avg_degree;

  // Check GetGeometricAvgDegreeColumn with conversion
  geometric_avg_degree =
      feature.GetGeometricAvgDegreeColumn(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*geometric_avg_degree, geometric_avg_in_degrees, 0.000001);
  delete geometric_avg_degree;
  EXPECT_THROW(feature.GetGeometricAvgDegreeColumn(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetGeometricAvgDegreeColumn with conversion and cached
  auto geometric_avg_degree_format =
      feature.GetGeometricAvgDegreeColumnCached(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*std::get<1>(geometric_avg_degree_format), geometric_avg_in_degrees, 0.000001);
  delete std::get<1>(geometric_avg_degree_format);

  auto cached_data = std::get<0>(geometric_avg_degree_format);
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
              geometric_avg_in_degrees, 0.000001);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
              geometric_avg_in_degrees, 0.000001);
}