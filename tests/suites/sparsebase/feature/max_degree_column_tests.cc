#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/max_degree_column.h"
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

class MaxDegreeColumnTest : public ::testing::Test {
 protected:
  feature::MaxDegreeColumn<int, int, int> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(MaxDegreeColumnTest, AllTests) {
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

  // Check GetMaxDegreeCSR implementation function
  Params1 p1;
  auto max_degree =
      feature::MaxDegreeColumn<int, int, int>::GetMaxDegreeColumnCSC({&global_csc}, &p1);

  auto max_in_degrees = 0;
  for (int i = 0; i < n; ++i)
    max_in_degrees = std::max(max_in_degrees, degrees[i]);
  EXPECT_EQ(*max_degree, max_in_degrees);
  delete max_degree;
  // Check GetMaxDegree
  max_degree = feature.GetMaxDegreeColumn(&global_csc, {&cpu_context}, true);
  EXPECT_EQ(*max_degree, max_in_degrees);
  delete max_degree;

  max_degree = feature.GetMaxDegreeColumn(&global_csc, {&cpu_context}, false);
  EXPECT_EQ(*max_degree, max_in_degrees);
  delete max_degree;

  // Check GetMaxDegree with conversion
  max_degree = feature.GetMaxDegreeColumn(&global_coo, {&cpu_context}, true);
  EXPECT_EQ(*max_degree, max_in_degrees);

  EXPECT_THROW(feature.GetMaxDegreeColumn(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check Extract
  auto feature_map = feature.Extract(&global_csc, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int *>(feature_map[feature.get_id()]), max_in_degrees);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }

  EXPECT_EQ(*std::any_cast<int *>(feature_map[feature.get_id()]), max_in_degrees);

  EXPECT_THROW(feature.Extract(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  delete max_degree;
}
