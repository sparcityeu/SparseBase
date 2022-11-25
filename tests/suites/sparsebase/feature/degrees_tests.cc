#include <memory>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/degrees.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::feature;
#include "../functionality_common.inc"

class DegreesTest : public ::testing::Test {
 protected:
  feature::Degrees<int, int, int> feature;

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
      feature::Degrees<int, int, int>::GetDegreesCSR({&global_csr}, &p1);
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
