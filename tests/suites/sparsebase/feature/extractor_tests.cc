#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/feature/feature_extractor.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"

#include "common.inc"

TEST(feature, ExtractorAdd) {
Extractor extractor =
    FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
extractor.Add(
    feature::Feature(Degrees<vertex_type, edge_type, value_type>{}));

vector<type_index> list = extractor.GetList();

string v1 =
    Degrees<vertex_type, edge_type, value_type>::get_id_static()
        .name();
string v2 = list[0].name();
EXPECT_EQ(v1, v2);
}

TEST(feature, ExtractorAddSub) {
Extractor extractor =
    FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
extractor.Add(Feature(Degrees<vertex_type, edge_type, value_type>{}));

vector<type_index> list = extractor.GetList();
EXPECT_EQ(list.size(), 1);

string v1 =
    Degrees<vertex_type, edge_type, value_type>::get_id_static()
        .name();
string v2 = list[0].name();
EXPECT_EQ(v1, v2);

extractor.Add(
    Feature(Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
            feature_type>{}));
list = extractor.GetList();
EXPECT_EQ(list.size(), 2);

extractor.Subtract(Feature(
    DegreeDistribution<vertex_type, edge_type, value_type, feature_type>{}));
extractor.Subtract(Feature(Degrees<vertex_type, edge_type, value_type>{}));
list = extractor.GetList();
EXPECT_EQ(list.size(), 0);
}

TEST_F(COOMock, ExtractorExtract) {
CPUContext cpu_context;
Extractor engine =
    FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
engine.Add(Feature(Degrees<vertex_type, edge_type, value_type>{}));

auto raw = engine.Extract(coo, {&cpu_context}, true);
EXPECT_EQ(raw.size(), 1);
auto degrees = any_cast<vertex_type *>(
    raw[Degrees<vertex_type, edge_type,
        value_type>::get_id_static()]);

EXPECT_THROW(engine.Extract(coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
engine.Add(Feature(
    DegreeDistribution<vertex_type, edge_type, value_type, feature_type>{}));
auto raw2 = engine.Extract(coo, {&cpu_context}, true);
EXPECT_EQ(raw2.size(), 2);
auto degree_dist = any_cast<feature_type *>(
    raw2[DegreeDistribution<vertex_type, edge_type, value_type,
        feature_type>::get_id_static()]);
EXPECT_EQ(degree_dist[4], 0);
EXPECT_THROW(engine.Extract(coo, {&cpu_context}, false),
utils::DirectExecutionNotAvailableException<
std::vector<std::type_index>>);
}

TEST_F(COOMock, ExtractorExtractGiven) {
CPUContext cpu_context;

vector<Feature> features;
features.push_back(Feature(Degrees<vertex_type, edge_type, value_type>{}));
features.push_back(Feature(
    DegreeDistribution<vertex_type, edge_type, value_type, feature_type>{}));
features.push_back(
    Feature(Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
            feature_type>{}));

auto results = Extractor::Extract(features, coo, {&cpu_context}, true);
EXPECT_EQ(results.size(), 2);
EXPECT_THROW(Extractor::Extract(features, coo, {&cpu_context}, false),
    utils::DirectExecutionNotAvailableException<
    std::vector<std::type_index>>);
}

TEST(feature, Feature) {
Degrees<vertex_type, edge_type, feature_type> degrees;
Feature f(degrees);
EXPECT_EQ(f->get_id(), degrees.get_id());
DegreeDistribution<vertex_type, edge_type, value_type, feature_type>
    degree_dist;
f = Feature(degree_dist);
EXPECT_EQ(f->get_id(), degree_dist.get_id());
Feature ff(std::move(f));
EXPECT_EQ(ff->get_id(), degree_dist.get_id());

const Feature fff(degrees);
Feature ffff(fff);
EXPECT_EQ(ffff->get_id(), degrees.get_id());
}
