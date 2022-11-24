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
#include "sparsebase/feature/degrees_degree_distribution.h"
#include "sparsebase/feature/degrees.h"
#include "sparsebase/feature/degree_distribution.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";

using namespace sparsebase;
using namespace sparsebase::preprocess;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
using namespace sparsebase::feature;
#include "../functionality_common.inc"

class Degrees_DegreeDistributionTest : public ::testing::Test {
 protected:
  Degrees_DegreeDistribution<int, int, int, float> feature;

  struct Params1 : sparsebase::utils::Parameters {};
  struct Params2 : sparsebase::utils::Parameters {};
};

TEST_F(Degrees_DegreeDistributionTest, FeaturePreprocessTypeTests) {
std::shared_ptr<Params1> p1(new Params1);
std::shared_ptr<Params2> p2(new Params2);
// Check getting feature id
EXPECT_EQ(
    std::type_index(typeid(Degrees_DegreeDistribution<int, int, int, float>)),
    feature.get_id());
// Getting a params object for an unset params
EXPECT_THROW(feature.get_params(std::type_index(typeid(int))).get(),
    utils::FeatureParamsException);
// Checking setting params of a sub-feature
feature.set_params(feature::Degrees<int, int, int>::get_id_static(), p1);
EXPECT_EQ(
    feature.get_params(feature::Degrees<int, int, int>::get_id_static()).get(),
    p1.get());
EXPECT_NE(
    feature.get_params(feature::Degrees<int, int, int>::get_id_static()).get(),
    p2.get());
// Checking setting params of feature that isn't a sub-feature
EXPECT_THROW(feature.set_params(typeid(int), p1),
utils::FeatureParamsException);
}
TEST_F(Degrees_DegreeDistributionTest, Degree_DegreeDistributionTests) {
// test get_sub_ids
EXPECT_EQ(feature.get_sub_ids().size(), 2);
std::vector<std::type_index> ids = {
    feature::Degrees<int, int, int>::get_id_static(),
    feature::DegreeDistribution<int, int, int, float>::get_id_static()};
std::sort(ids.begin(), ids.end());
EXPECT_EQ(feature.get_sub_ids()[0], ids[0]);
EXPECT_EQ(feature.get_sub_ids()[1], ids[1]);

// Test get_subs
auto subs = feature.get_subs();
// two sub-feature
EXPECT_EQ(subs.size(), 2);
// same type as feature but different address
auto &feat = *(subs[0]);
EXPECT_EQ(std::type_index(typeid(feat)), ids[0]);
auto &feat1 = *(subs[1]);
EXPECT_EQ(std::type_index(typeid(feat1)), ids[1]);
EXPECT_NE(subs[0], &feature);
EXPECT_NE(subs[1], &feature);

// Check GetCSR implementation function
Params1 p1;
auto degrees_and_distribution_map =
    Degrees_DegreeDistribution<int, int, int, float>::GetCSR({&global_csr},
                                                             &p1);
ASSERT_EQ(degrees_and_distribution_map.size(), 2);
ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
    degrees_and_distribution_map.end());
ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
    degrees_and_distribution_map.end());
ASSERT_NO_THROW(std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]));
auto distribution_array = std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]);
ASSERT_NO_THROW(std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]));
auto degree_array = std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
EXPECT_EQ(degree_array[i], degrees[i]);
}
delete[] distribution_array;
delete[] degree_array;
//// Check Get (function matcher)
degrees_and_distribution_map = feature.Get(&global_csr, {&cpu_context}, true);
ASSERT_EQ(degrees_and_distribution_map.size(), 2);
ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
    degrees_and_distribution_map.end());
ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
    degrees_and_distribution_map.end());
ASSERT_NO_THROW(std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]));
distribution_array = std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]);
ASSERT_NO_THROW(std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]));
degree_array = std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
EXPECT_EQ(degree_array[i], degrees[i]);
}
delete[] distribution_array;
delete[] degree_array;
//// Check Get with conversion (function matcher)
degrees_and_distribution_map = feature.Get(&global_coo, {&cpu_context}, true);
ASSERT_EQ(degrees_and_distribution_map.size(), 2);
ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
    degrees_and_distribution_map.end());
ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
    degrees_and_distribution_map.end());
ASSERT_NO_THROW(std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]));
distribution_array = std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]);
ASSERT_NO_THROW(std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]));
degree_array = std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
EXPECT_EQ(degree_array[i], degrees[i]);
}
delete[] distribution_array;
delete[] degree_array;
EXPECT_THROW(feature.Get(&global_coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
// Check Extract
degrees_and_distribution_map =
feature.Extract(&global_csr, {&cpu_context}, true);
ASSERT_EQ(degrees_and_distribution_map.size(), 2);
ASSERT_NE(degrees_and_distribution_map.find(ids[0]),
    degrees_and_distribution_map.end());
ASSERT_NE(degrees_and_distribution_map.find(ids[1]),
    degrees_and_distribution_map.end());
ASSERT_NO_THROW(std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]));
distribution_array = std::any_cast<float *>(
    degrees_and_distribution_map
    [feature::DegreeDistribution<int, int, int, float>::get_id_static()]);
ASSERT_NO_THROW(std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]));
degree_array = std::any_cast<int *>(
    degrees_and_distribution_map[feature::Degrees<int, int,
        int>::get_id_static()]);
for (int i = 0; i < n; i++) {
EXPECT_EQ(distribution_array[i], distribution[i]);
EXPECT_EQ(degree_array[i], degrees[i]);
}
delete[] distribution_array;
delete[] degree_array;
}

