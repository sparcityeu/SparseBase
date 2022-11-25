//
// Created by Taha Atahan Akyildiz on 23.10.2022.
//

#include <iostream>

#include "gtest/gtest.h"
#include "sparsebase/sparsebase.h"

using namespace std;
using namespace sparsebase;
using namespace format;
using namespace experiment;
using namespace utils;
using namespace io;
;
using namespace context;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

const string FILE_NAME = "../../../../examples/data/ash958.mtx";
auto load_data_f(vector<string>& file_names) {
  cout << FILE_NAME << endl;
  CSR<vertex_type, edge_type, value_type>* csr =
      MTXReader<vertex_type, edge_type, value_type>(FILE_NAME).ReadCSR();
  unordered_map<string, Format*> r;
  r.emplace("format", csr);
  return r;
};

void preprocess_f(unordered_map<string, Format*>& data, std::any& fparams,
                  std::any& params) {
  data.emplace("new_format", data["format"]);
};

std::any kernel_f(unordered_map<string, Format*>& data, std::any& fparams,
                  std::any& pparams, std::any& kparams) {
  return {};
};

class ExperimentFixture : public ::testing::Test {
 protected:
  ConcreteExperiment exp;
  std::any fparams;
  vector<string> files = {FILE_NAME};

  void SetUp() {
    exp.AddDataLoader(load_data_f, {make_pair(files, fparams)});
    exp.AddPreprocess("pid", preprocess_f, {});
    exp.AddKernel("kid", kernel_f, {});
  }
};

TEST_F(ExperimentFixture, basics) {
  exp.Run(2, true);
  auto aux = exp.GetAuxiliary();
  EXPECT_EQ(aux.size(), 2);
  auto res = exp.GetResults();
  EXPECT_EQ(res.size(), 2);
  auto rtimes = exp.GetRunTimes();
  EXPECT_EQ(rtimes.size(), 2);
}

TEST_F(ExperimentFixture, basics2) {
  exp.Run();
  auto aux = exp.GetAuxiliary();
  EXPECT_EQ(aux.size(), 0);
  auto res = exp.GetResults();
  EXPECT_EQ(res.size(), 1);
  auto rtimes = exp.GetRunTimes();
  EXPECT_EQ(rtimes.size(), 1);
}

TEST_F(ExperimentFixture, identifiers) {
  exp.Run(3, true);
  vector<string> aux_identifiers;
  vector<string> res_identifiers;

  for (const auto& file : files) {
    aux_identifiers.push_back("format,-" + file);
    aux_identifiers.push_back("new_format,-" + file + ",pid");
    for (int i = 0; i < 3; i++) {
      res_identifiers.push_back("-" + file + ",pid,kid," + to_string(i));
    }
  }

  auto aux = exp.GetAuxiliary();
  EXPECT_EQ(aux_identifiers.size(), aux.size());
  for (const auto& i : aux_identifiers) {
    cout << i << endl;
    EXPECT_TRUE(aux.find(i) != aux.end());
  }

  auto res = exp.GetResults();
  EXPECT_EQ(res_identifiers.size(), res.size());
  auto rtimes = exp.GetRunTimes();
  EXPECT_EQ(res_identifiers.size(), rtimes.size());
  for (const auto& i : res_identifiers) {
    cout << i << endl;
    EXPECT_TRUE(res.find(i) != res.end());
    EXPECT_TRUE(rtimes.find(i) != rtimes.end());
  }
}

TEST(Experiment, incomplete) {
  ConcreteExperiment exp;
  std::any fparams;
  vector<string> files = {FILE_NAME};
  exp.AddDataLoader(load_data_f, {make_pair(files, fparams)});
  exp.Run(3, true);
  auto aux = exp.GetAuxiliary();
  EXPECT_EQ(aux.size(), 1);
  exp.AddPreprocess("id", preprocess_f, {});
  exp.Run(3);
  aux = exp.GetAuxiliary();
  EXPECT_EQ(aux.size(), 1);
  exp.Run(3, true);
  aux = exp.GetAuxiliary();
  EXPECT_EQ(aux.size(), 2);
}
