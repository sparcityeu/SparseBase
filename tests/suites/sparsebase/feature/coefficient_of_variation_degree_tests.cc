#include <memory>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/context/context.h"
#include "sparsebase/feature/coefficient_of_variation_degree.h"
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

class CoefficientOfVariationDegreeTest : public ::testing::Test {
 protected:
  CoefficientOfVariationDegree<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(CoefficientOfVariationDegreeTest, AllTests) {
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

  // Check GetCoefficientOfVariationDegreeCSR implementation function
  Params1 p1;
  auto coefficient_of_variation_degree =
      CoefficientOfVariationDegree<int, int, int, float>::GetCoefficientOfVariationDegreeCSR(
          {&global_csr}, &p1);

  int degree_sum = row_ptr[n] - row_ptr[0];
  auto avg_in_degrees = degree_sum / (float) n;
  auto standard_deviation_in_degrees = 0.0;
  for (int i = 0; i < n; i++) {
    standard_deviation_in_degrees += (row_ptr[i + 1] - row_ptr[i] - avg_in_degrees)*(row_ptr[i + 1] - row_ptr[i] - avg_in_degrees);
  }
  auto coefficient_of_variation_in_degrees = sqrt(standard_deviation_in_degrees) / avg_in_degrees;

  EXPECT_NEAR(*coefficient_of_variation_degree, coefficient_of_variation_in_degrees, 0.000001);

  delete coefficient_of_variation_degree;
  //// Check GetCoefficientOfVariationDegree (function matcher)
  coefficient_of_variation_degree =
      feature.GetCoefficientOfVariationDegree(&global_csr, {&cpu_context}, true);
  EXPECT_NEAR(*coefficient_of_variation_degree, coefficient_of_variation_in_degrees, 0.000001);
  delete coefficient_of_variation_degree;

  coefficient_of_variation_degree =
      feature.GetCoefficientOfVariationDegree(&global_csr, {&cpu_context}, false);
  EXPECT_NEAR(*coefficient_of_variation_degree, coefficient_of_variation_in_degrees, 0.000001);
  delete coefficient_of_variation_degree;

  // Check GetCoefficientOfVariationDegree with conversion
  coefficient_of_variation_degree =
      feature.GetCoefficientOfVariationDegree(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*coefficient_of_variation_degree, coefficient_of_variation_in_degrees, 0.000001);
  delete coefficient_of_variation_degree;
  EXPECT_THROW(feature.GetCoefficientOfVariationDegree(&global_coo, {&cpu_context}, false),
               utils::DirectExecutionNotAvailableException<
                   std::vector<std::type_index>>);
  // Check GetCoefficientOfVariationDegree with conversion and cached
  auto coefficient_of_variation_degree_format =
      feature.GetCoefficientOfVariationDegreeCached(&global_coo, {&cpu_context}, true);
  EXPECT_NEAR(*std::get<1>(coefficient_of_variation_degree_format), coefficient_of_variation_in_degrees, 0.000001);
  delete std::get<1>(coefficient_of_variation_degree_format);

  auto cached_data = std::get<0>(coefficient_of_variation_degree_format);
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
            coefficient_of_variation_in_degrees, 0.000001);

  // Check Extract with conversion
  feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
  // Check map size and type
  EXPECT_EQ(feature_map.size(), 1);
  for (auto feat : feature_map) {
    EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
  }
  EXPECT_NEAR(*std::any_cast<float *>(feature_map[feature.get_id()]),
            coefficient_of_variation_in_degrees, 0.000001);            
}
