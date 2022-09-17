//
// Created by Taha Atahan Akyildiz on 17.04.2022.
//
#include <iostream>

#include "sparsebase/feature/feature.h"
#include "sparsebase/format/format.h"
#include "gtest/gtest.h"

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = float;
using feature_type = float;

using namespace std;

using namespace sparsebase;
using namespace format;
using namespace preprocess;
using namespace feature;
using namespace context;

class COOMock : public ::testing::Test {
protected:
  COO<vertex_type, edge_type, value_type> *coo;
  vertex_type rows[6] = {0, 0, 1, 1, 2, 3};
  edge_type cols[6] = {0, 1, 1, 2, 3, 3};
  value_type vals[6] = {10, 20, 30, 40, 50, 60};

  void SetUp() override {
    coo = new COO<vertex_type, edge_type, value_type>(6, 6, 6, rows, cols, vals,
                                                      kNotOwned);
  }
  void TearDown() override { delete coo; }
};

TEST(feature, ExtractorAdd) {
  Extractor extractor =
      FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
  extractor.Add(
      feature::Feature(Degrees<vertex_type, edge_type, value_type>{}));

  vector<type_index> list = extractor.GetList();

  string v1 =
      Degrees<vertex_type, edge_type, value_type>::get_feature_id_static()
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
      Degrees<vertex_type, edge_type, value_type>::get_feature_id_static()
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

TEST(feature, FeatureExtractorRegisterMap) {
  std::vector<type_index> expected = {
      DegreeDistribution<vertex_type, edge_type, value_type,
                         feature_type>::get_feature_id_static(),
      Degrees<vertex_type, edge_type, feature_type>::get_feature_id_static(),
      Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
                                 feature_type>::get_feature_id_static()};
  sort(expected.begin(), expected.end());
  Extractor extractor =
      FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
  vector<preprocess::ExtractableType *> featFuncs = extractor.GetFuncList();
  vector<type_index> funcs;
  for (auto &func : featFuncs) {
    funcs.push_back(func->get_feature_id());
    // std::cout << func->get_feature_id().name() << endl;
  }
  sort(funcs.begin(), funcs.end());
  EXPECT_EQ(funcs.size(), expected.size());
  for (unsigned int i = 0; i < funcs.size(); i++) {
    EXPECT_EQ(funcs[i].name(), expected[i].name());
  }
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
                  value_type>::get_feature_id_static()]);

  EXPECT_THROW(engine.Extract(coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
  engine.Add(Feature(
      DegreeDistribution<vertex_type, edge_type, value_type, feature_type>{}));
  auto raw2 = engine.Extract(coo, {&cpu_context}, true);
  EXPECT_EQ(raw2.size(), 2);
  auto degree_dist = any_cast<feature_type *>(
      raw2[DegreeDistribution<vertex_type, edge_type, value_type,
                              feature_type>::get_feature_id_static()]);
  EXPECT_EQ(degree_dist[4], 0);
  EXPECT_THROW(engine.Extract(coo, {&cpu_context}, false),  utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
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
  EXPECT_THROW(Extractor::Extract(features, coo, {&cpu_context}, false), utils::DirectExecutionNotAvailableException<std::vector<std::type_index>>);
}

// ClassMatcherMixin functions are protected
// TEST(feature, ClassMatcherMixinMatch){
//   ClassMatcherMixin<ExtractableType *> cmm;
//   auto degree_distribution =
//       new DegreeDistribution<vertex_type, edge_type, value_type,
//       feature_type>();
//   cmm.RegisterClass(degree_distribution->get_sub_ids(), degree_distribution);
//   auto degrees = new Degrees<vertex_type, edge_type, feature_type>();
//   cmm.RegisterClass(degrees->get_sub_ids(), degrees);
//   auto degrees_degreedistribution =
//       new Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
//       feature_type>();
//   cmm.RegisterClass(degrees_degreedistribution->get_sub_ids(),
//                       degrees_degreedistribution);
//
//   unordered_map<type_index, ExtractableType *> in;
//   in.insert(make_pair(DegreeDistribution<vertex_type, edge_type, value_type,
//   feature_type>::get_feature_id_static(), new DegreeDistribution<vertex_type,
//   edge_type, value_type, feature_type>()));
//   in.insert(make_pair(Degrees<vertex_type, edge_type,
//   feature_type>::get_feature_id_static(), new Degrees<vertex_type, edge_type,
//   feature_type>()));
//
//   vector<ExtractableType*> rs = cmm.GetClasses(in);
//   EXPECT_EQ(rs.size(), 1);
//   EXPECT_EQ(rs[0]->get_feature_id(),
//   degrees_degreedistribution->get_feature_id());
//
//   in.insert(make_pair(Degrees_DegreeDistribution<vertex_type, edge_type,
//   value_type, feature_type>::get_feature_id_static(), new
//   Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
//   feature_type>())); rs = cmm.GetClasses(in); EXPECT_EQ(rs.size(), 2);
//   EXPECT_EQ(rs[0]->get_feature_id(),
//   degrees_degreedistribution->get_feature_id());
//   EXPECT_EQ(rs[1]->get_feature_id(),
//   degrees_degreedistribution->get_feature_id());
// }

TEST(feature, Feature) {
  Degrees<vertex_type, edge_type, feature_type> degrees;
  Feature f(degrees);
  EXPECT_EQ(f->get_feature_id(), degrees.get_feature_id());
  DegreeDistribution<vertex_type, edge_type, value_type, feature_type>
      degree_dist;
  f = Feature(degree_dist);
  EXPECT_EQ(f->get_feature_id(), degree_dist.get_feature_id());
  Feature ff(std::move(f));
  EXPECT_EQ(ff->get_feature_id(), degree_dist.get_feature_id());

  const Feature fff(degrees);
  Feature ffff(fff);
  EXPECT_EQ(ffff->get_feature_id(), degrees.get_feature_id());
}