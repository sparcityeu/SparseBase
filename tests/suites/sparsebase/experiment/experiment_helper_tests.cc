#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/reorder/reorderer.h"
#include "sparsebase/sparsebase.h"

using namespace std;
using namespace sparsebase;
using namespace format;
using namespace experiment;
using namespace utils;
using namespace io;
;
using namespace reorder;
using namespace context;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

const string FILE_NAME = "../../../../examples/data/ash958.mtx";
class LoadDataFixture : public ::testing::Test {
 protected:
  std::any fparams;
  vector<string> files = {FILE_NAME};
};

TEST_F(LoadDataFixture, LoadCSR) {
  auto data = LoadCSR<MTXReader, vertex_type, edge_type, value_type>(files);
  EXPECT_EQ(data.size(), 1);
  EXPECT_TRUE(data.find("format") != data.end());
  bool res = data["format"]->Is<CSR<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

TEST_F(LoadDataFixture, LoadCOO) {
  auto data = LoadCOO<MTXReader, vertex_type, edge_type, value_type>(files);
  EXPECT_EQ(data.size(), 1);
  EXPECT_TRUE(data.find("format") != data.end());
  bool res = data["format"]->Is<COO<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

TEST_F(LoadDataFixture, LoadCSC) {
  auto data = LoadCSC<MTXReader, vertex_type, edge_type, value_type>(files);
  EXPECT_EQ(data.size(), 1);
  EXPECT_TRUE(data.find("format") != data.end());
  bool res = data["format"]->Is<CSC<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

TEST_F(LoadDataFixture, LoadFormat) {
  auto data =
      LoadFormat<CSR, MTXReader, vertex_type, edge_type, value_type>(files);
  EXPECT_EQ(data.size(), 1);
  EXPECT_TRUE(data.find("format") != data.end());
  bool res = data["format"]->Is<CSR<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

class ReorderFixture : public ::testing::Test {
 protected:
  std::any fparams;
  vector<string> files = {FILE_NAME};
  std::any p1;
  unordered_map<string, Format*> data;

  void SetUp() {
    data =
        LoadFormat<CSR, MTXReader, vertex_type, edge_type, value_type>(files);
  }
  void TearDown() {}
};

TEST_F(ReorderFixture, ReorderCSR) {
  RCMReorder<vertex_type, edge_type, value_type>::ParamsType p2 = {};
  ReorderCSR<RCMReorder, CPUContext, vertex_type, edge_type, value_type>(
      data, p1, p2);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(data.find("processed_format") != data.end());
  bool res =
      data["processed_format"]->Is<CSR<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

TEST_F(ReorderFixture, Reorder) {
  RCMReorder<vertex_type, edge_type, value_type>::ParamsType p2 = {};
  Reorder<RCMReorder, CSR, CPUContext, vertex_type, edge_type, value_type>(
      data, p1, p2);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(data.find("processed_format") != data.end());
  bool res =
      data["processed_format"]->Is<CSR<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}

TEST_F(ReorderFixture, Pass) {
  std::any p2 = {};
  Pass(data, p1, p2);
  EXPECT_EQ(data.size(), 2);
  EXPECT_TRUE(data.find("processed_format") != data.end());
  bool res =
      data["processed_format"]->Is<CSR<vertex_type, edge_type, value_type>>();
  EXPECT_TRUE(res);
}
