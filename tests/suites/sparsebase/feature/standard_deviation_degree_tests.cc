#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/standard_deviation_degree.h"
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

class StandardDeviationDegreeTest : public ::testing::Test {
 protected:
  StandardDeviationDegree<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(StandardDeviationDegreeTest, AllTests) {
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

  // Check GetStandardDeviationDegreeCSR implementation function
  Params1 p1;
  auto standard_deviation_degree =
      StandardDeviationDegree<int, int, int, float>::GetStandardDeviationDegreeCSR(
          {&global_csr}, &p1);

  int degree_sum = row_ptr[n] - row_ptr[0];
  auto avg_in_degrees = degree_sum / (float) n;
  auto standard_deviation_in_degrees = 0.0;
  for (int i = 0; i < n; i++) {
    standard_deviation_in_degrees += (row_ptr[i + 1] - row_ptr[i] - avg_in_degrees)*(row_ptr[i + 1] - row_ptr[i] - avg_in_degrees);
  }
  standard_deviation_in_degrees = sqrt(standard_deviation_in_degrees);

  EXPECT_NEAR(*standard_deviation_degree, standard_deviation_in_degrees, 0.000001);

  delete standard_deviation_degree;
  //// Check GetStandardDeviationDegree (function matcher)
  standard_deviation_degree =
      feature.GetStandardDeviationDegree(&global_csr, {&cpu_context}, true);
  EXPECT_NEAR(*standard_deviation_degree, standard_deviation_in_degrees, 0.000001);
  delete standard_deviation_degree;

  standard_deviation_degree =
      feature.GetStandardDeviationDegree(&global_csr, {&cpu_context}, false);
  EXPECT_NEAR(*standard_deviation_degree, standard_deviation_in_degrees, 0.000001);
  delete standard_deviation_degree;

  // Check GetStandardDeviationDegree with conversion
  standard_deviation_degree =
      feature.GetStandardDeviationDegree(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*standard_deviation_degree, standard_deviation_in_degrees, 0.000001);
  delete standard_deviation_degree;
  EXPECT_THROW(feature.GetStandardDeviationDegree(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetStandardDeviationDegree with conversion and cached
  auto standard_deviation_degree_format =
      feature.GetStandardDeviationDegreeCached(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*std::get<1>(standard_deviation_degree_format), standard_deviation_in_degrees, 0.000001);
  delete std::get<1>(standard_deviation_degree_format);

  auto cached_data = std::get<0>(standard_deviation_degree_format);
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
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
            standard_deviation_in_degrees, 0.000001);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
            standard_deviation_in_degrees, 0.000001);            
}
