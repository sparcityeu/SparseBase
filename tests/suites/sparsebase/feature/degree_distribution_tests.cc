#include <iostream>
#include <memory>
#include <set>
#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "sparsebase/bases/reorder_base.h"
#include "sparsebase/config.h"
#include "sparsebase/context/context.h"
#include "sparsebase/converter/converter.h"
#include "sparsebase/format/coo.h"
#include "sparsebase/format/csc.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/reorder/degree_reorder.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"
#include "sparsebase/feature/degree_distribution.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::feature;
#include "../functionality_common.inc"

class DegreeDistributionTest : public ::testing::Test {
 protected:
  DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(DegreeDistributionTest, AllTests) {
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

// Check GetDegreeDistributionCSR implementation function
Params1 p1;
auto distribution_array =
    DegreeDistribution<int, int, int, float>::GetDegreeDistributionCSR(
        {&global_csr}, &p1);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
}
delete[] distribution_array;
//// Check GetDistribution (function matcher)
distribution_array =
feature.GetDistribution(&global_csr, {&cpu_context}, true);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
}
delete[] distribution_array;
distribution_array =
feature.GetDistribution(&global_csr, {&cpu_context}, false);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
}
delete[] distribution_array;
// Check GetDistribution with conversion
distribution_array =
feature.GetDistribution(&global_coo, {&cpu_context}, true);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
}
delete[] distribution_array;
EXPECT_THROW(feature.GetDistribution(&global_coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
// Check GetDistribution with conversion and cached
auto distribution_array_format =
    feature.GetDistributionCached(&global_coo, {&cpu_context}, true);
for (int i = 0; i < n; i++) {
EXPECT_EQ(std::get<1>(distribution_array_format)[i], distribution[i]);
}
delete[] std::get<1>(distribution_array_format);
auto cached_data = std::get<0>(distribution_array_format);
ASSERT_EQ(cached_data.size(), 1);
ASSERT_EQ(cached_data[0][0]->get_id(),
    std::type_index(typeid(global_csr)));
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
for (int i = 0; i < n; i++) {
EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_id()])[i],
distribution[i]);
}
// Check Extract with conversion
feature_map = feature.Extract(&global_coo, {&cpu_context}, true);
// Check map size and type
EXPECT_EQ(feature_map.size(), 1);
for (auto feat : feature_map) {
EXPECT_EQ(feat.first, std::type_index(typeid(feature)));
}
for (int i = 0; i < n; i++) {
EXPECT_EQ(std::any_cast<float *>(feature_map[feature.get_id()])[i],
distribution[i]);
}
}
