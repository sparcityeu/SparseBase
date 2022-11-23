//
// Created by Taha Atahan Akyildiz on 17.04.2022.
//
#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/feature/feature_extractor.h"
#include "sparsebase/format/csr.h"
#include "sparsebase/format/format.h"
#include "sparsebase/format/format_order_one.h"
#include "sparsebase/format/format_order_two.h"
#include "sparsebase/utils/extractable.h"

#include "common.inc"


TEST(feature, FeatureExtractorRegisterMap) {
  std::vector<type_index> expected = {
      DegreeDistribution<vertex_type, edge_type, value_type,
                         feature_type>::get_id_static(),
      Degrees<vertex_type, edge_type, feature_type>::get_id_static(),
      Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
                                 feature_type>::get_id_static()};
  sort(expected.begin(), expected.end());
  Extractor extractor =
      FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
  vector<utils::Extractable *> featFuncs = extractor.GetFuncList();
  vector<type_index> funcs;
  for (auto &func : featFuncs) {
    funcs.push_back(func->get_id());
    // std::cout << func->get_id().name() << endl;
  }
  sort(funcs.begin(), funcs.end());
  EXPECT_EQ(funcs.size(), expected.size());
  for (unsigned int i = 0; i < funcs.size(); i++) {
    EXPECT_EQ(funcs[i].name(), expected[i].name());
  }
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
//   feature_type>::get_id_static(), new DegreeDistribution<vertex_type,
//   edge_type, value_type, feature_type>()));
//   in.insert(make_pair(Degrees<vertex_type, edge_type,
//   feature_type>::get_id_static(), new Degrees<vertex_type, edge_type,
//   feature_type>()));
//
//   vector<ExtractableType*> rs = cmm.GetClasses(in);
//   EXPECT_EQ(rs.size(), 1);
//   EXPECT_EQ(rs[0]->get_id(),
//   degrees_degreedistribution->get_id());
//
//   in.insert(make_pair(Degrees_DegreeDistribution<vertex_type, edge_type,
//   value_type, feature_type>::get_id_static(), new
//   Degrees_DegreeDistribution<vertex_type, edge_type, value_type,
//   feature_type>())); rs = cmm.GetClasses(in); EXPECT_EQ(rs.size(), 2);
//   EXPECT_EQ(rs[0]->get_id(),
//   degrees_degreedistribution->get_id());
//   EXPECT_EQ(rs[1]->get_id(),
//   degrees_degreedistribution->get_id());
// }
