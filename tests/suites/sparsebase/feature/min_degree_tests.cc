#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/min_degree.h"
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

class MinDegreeTest : public ::testing::Test {
 protected:
  feature::MinDegree<int, int, int> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(MinDegreeTest, AllTests) {
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

  // Check GetMinDegreeCSR implementation function
  Params1 p1;
  auto min_degree =
      feature::MinDegree<int, int, int>::GetMinDegreeCSR({&global_csr}, &p1);

  auto min_in_degrees = (int)1e9;
  for (int i = 0; i < n; ++i)
    min_in_degrees = std::min(min_in_degrees, degrees[i]);
  EXPECT_EQ(*min_degree, min_in_degrees);
  delete min_degree;
  // Check GetMinDegree
  min_degree = feature.GetMinDegree(&global_csr, {&cpu_context}, true);
  EXPECT_EQ(*min_degree, min_in_degrees);
  delete min_degree;

  min_degree = feature.GetMinDegree(&global_csr, {&cpu_context}, false);
  EXPECT_EQ(*min_degree, min_in_degrees);
  delete min_degree;

  // Check GetMinDegree with conversion
  min_degree = feature.GetMinDegree(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*min_degree, min_in_degrees);

  EXPECT_THROW(feature.GetMinDegree(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check Extract
  auto feature_map = feature.Extract(&global_csr, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int *>(feature_map[feature.get_id()]), min_in_degrees);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int *>(feature_map[feature.get_id()]), min_in_degrees);

  EXPECT_THROW(feature.Extract(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  delete min_degree;
}
