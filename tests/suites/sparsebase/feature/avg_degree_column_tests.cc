#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/avg_degree_column.h"
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

class AvgDegreeColumnTest : public ::testing::Test {
 protected:
  AvgDegreeColumn<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(AvgDegreeColumnTest, AllTests) {
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

  // Check GetAvgDegreeColumnCSC implementation function
  Params1 p1;
  auto avg_degree_column =
      AvgDegreeColumn<int, int, int, float>::GetAvgDegreeColumnCSC(
          {&global_csc}, &p1);

  int degree_sum = col_ptr[n] - col_ptr[0];
  auto avg_in_degrees = degree_sum / (float) n;

  EXPECT_EQ(*avg_degree_column, avg_in_degrees);

  delete avg_degree_column;
  //// Check GetAvgDegree (function matcher)
  avg_degree_column =
      feature.GetAvgDegreeColumn(&global_csc, {&cpu_context}, true);
  EXPECT_EQ(*avg_degree_column, avg_in_degrees);
  delete avg_degree_column;

  avg_degree_column =
      feature.GetAvgDegreeColumn(&global_csc, {&cpu_context}, false);
  EXPECT_EQ(*avg_degree_column, avg_in_degrees);
  delete avg_degree_column;

  // Check GetAvgDegree with conversion
  avg_degree_column =
      feature.GetAvgDegreeColumn(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*avg_degree_column, avg_in_degrees);
  delete avg_degree_column;
  EXPECT_THROW(feature.GetAvgDegreeColumn(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetAvgDegree with conversion and cached
  auto avg_degree_column_format =
      feature.GetAvgDegreeColumnCached(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*std::get<1>(avg_degree_column_format), avg_in_degrees);
  delete std::get<1>(avg_degree_column_format);

  auto cached_data = std::get<0>(avg_degree_column_format);
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
  EXPECT_EQ(*std::any_cast<float *>(feature_map[feature.get_id()]),
            avg_in_degrees);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_EQ(*std::any_cast<float *>(feature_map[feature.get_id()]),
            avg_in_degrees);
}
