#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/avg_degree.h"
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

class AvgDegreeTest : public ::testing::Test {
 protected:
  AvgDegree<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(AvgDegreeTest, AllTests) {
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

  // Check GetAvgDegreeCSR implementation function
  Params1 p1;
  auto avg_degree =
      AvgDegree<int, int, int, float>::GetAvgDegreeCSR(
          {&global_csr}, &p1);

  int degree_sum = row_ptr[n] - row_ptr[0];
  auto avg_in_degrees = degree_sum / (float) n;

  EXPECT_EQ(*avg_degree, avg_in_degrees);

  delete avg_degree;
  //// Check GetAvgDegree (function matcher)
  avg_degree =
      feature.GetAvgDegree(&global_csr, {&cpu_context}, true);
  EXPECT_EQ(*avg_degree, avg_in_degrees);
  delete avg_degree;

  avg_degree =
      feature.GetAvgDegree(&global_csr, {&cpu_context}, false);
  EXPECT_EQ(*avg_degree, avg_in_degrees);
  delete avg_degree;

  // Check GetAvgDegree with conversion
  avg_degree =
      feature.GetAvgDegree(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*avg_degree, avg_in_degrees);
  delete avg_degree;
  EXPECT_THROW(feature.GetAvgDegree(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetAvgDegree with conversion and cached
  auto avg_degree_format =
      feature.GetAvgDegreeCached(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*std::get<1>(avg_degree_format), avg_in_degrees);
  delete std::get<1>(avg_degree_format);

  auto cached_data = std::get<0>(avg_degree_format);
  ASSERT_EQ(cached_data.size(), 1);
  ASSERT_EQ(cached_data[0][0]->get_id(), std::type_index(typeid(global_csr)));
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
