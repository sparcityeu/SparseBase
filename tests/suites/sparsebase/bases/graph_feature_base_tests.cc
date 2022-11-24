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
#include "sparsebase/bases/graph_feature_base.h"
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
#include "sparsebase/reorder/generic_reorder.h"
#include "sparsebase/reorder/gray_reorder.h"
#include "sparsebase/reorder/reorder_heatmap.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/utils/exception.h"

const std::string FILE_NAME = "../../../../examples/data/ash958.mtx";


using namespace sparsebase;
using namespace sparsebase::preprocess;
using namespace sparsebase::reorder;
using namespace sparsebase::bases;
#include "../functionality_common.inc"
TEST(GraphFeatureBase, Degrees) {
EXPECT_NO_THROW(GraphFeatureBase::GetDegrees(
    &global_csr, {&cpu_context}, true));
auto degrees_array = GraphFeatureBase::GetDegrees(
    &global_csr, {&cpu_context}, true);
EXPECT_EQ(std::type_index(typeid(degrees_array)),
    std::type_index(typeid(int *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degrees_array[i], degrees[i]);
}
}
TEST(GraphFeatureBase, DegreesCached) {
EXPECT_NO_THROW(GraphFeatureBase::GetDegreesCached(
    &global_csr, {&cpu_context}));
auto output = GraphFeatureBase::GetDegreesCached(
    &global_csr, {&cpu_context});
auto degrees_array = output.second;
EXPECT_EQ(std::type_index(typeid(degrees_array)),
    std::type_index(typeid(int *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degrees_array[i], degrees[i]);
}
EXPECT_EQ(output.first.size(), 0);
auto output_conv = GraphFeatureBase::GetDegreesCached(
    &global_coo, {&cpu_context});
degrees_array = output_conv.second;
EXPECT_EQ(output_conv.first.size(), 1);
EXPECT_EQ(std::type_index(typeid(degrees_array)),
    std::type_index(typeid(int *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degrees_array[i], degrees[i]);
}
}

TEST(GraphFeatureBase, DegreeDistribution) {
EXPECT_NO_THROW(
    GraphFeatureBase::GetDegreeDistribution<float>(
    &global_csr, {&cpu_context}, true));
auto degreeDistribution_array =
    GraphFeatureBase::GetDegreeDistribution<float>(
        &global_csr, {&cpu_context}, true);
EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
    std::type_index(typeid(float *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
}
}
TEST(GraphFeatureBase, DegreeDistributionCached) {
EXPECT_NO_THROW(
    GraphFeatureBase::GetDegreeDistributionCached<
    float>(&global_csr, {&cpu_context}));
auto output =
    GraphFeatureBase::GetDegreeDistributionCached<
        float>(&global_csr, {&cpu_context});
auto degreeDistribution_array = output.second;
EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
    std::type_index(typeid(float *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
}
EXPECT_EQ(output.first.size(), 0);
auto output_conv =
    GraphFeatureBase::GetDegreeDistributionCached<
        float>(&global_coo, {&cpu_context});
EXPECT_EQ(output_conv.first.size(), 1);
degreeDistribution_array = output_conv.second;
EXPECT_EQ(std::type_index(typeid(degreeDistribution_array)),
    std::type_index(typeid(float *)));
for (int i = 0; i < n; i++) {
EXPECT_EQ(degreeDistribution_array[i], distribution[i]);
}
}
