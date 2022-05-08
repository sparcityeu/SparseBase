//
// Created by Taha Atahan Akyildiz on 17.04.2022.
//
#include "gtest/gtest.h"
#include "sparsebase/format/format.h"
#include "sparsebase/feature/feature.h"

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = float;
using feature_type = float;

using namespace std;

using namespace sparsebase;
using namespace format;
using namespace preprocess;
using namespace feature;

TEST(feature, ExtractorAdd){
  Extractor extractor = FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
  extractor.Add(feature::Feature(Degrees<vertex_type, edge_type, value_type>{}));

  vector<type_index> list = extractor.GetList();

  string v1 = Degrees<vertex_type, edge_type, value_type>::get_feature_id_static().name();
  string v2 = list[0].name();
  EXPECT_EQ(v1, v2);
}

TEST(feature, ExtractorAddSub){
  Extractor extractor = FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
  extractor.Add(Feature(Degrees<vertex_type, edge_type, value_type>{}));

  vector<type_index> list = extractor.GetList();
  EXPECT_EQ(list.size(), 1);

  string v1 = Degrees<vertex_type, edge_type, value_type>::get_feature_id_static().name();
  string v2 = list[0].name();
  EXPECT_EQ(v1, v2);

  extractor.Subtract(Feature(Degrees<vertex_type, edge_type, value_type>{}));
  list = extractor.GetList();
  EXPECT_EQ(list.size(), 0);
}
