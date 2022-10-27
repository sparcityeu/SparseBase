//
// Created by Taha Atahan Akyildiz on 23.10.2022.
//

#include "sparsebase/sparsebase.h"
#include "gtest/gtest.h"
#include <iostream>

using namespace std;
using namespace sparsebase;
using namespace format;
using namespace experiment;
using namespace utils;
using namespace io;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;

const ConcreteExperiment experiment;
const string FILE_NAME = "../../../../examples/data/ash958.mtx";

auto load_data_f( vector<string> & file_names ) {
  CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(FILE_NAME)
      .ReadCSR();
  unordered_map<string, Format*> r;
  r.emplace("format", csr);
  return r;
};

void preprocess_f(unordered_map<string, Format*> & data, std::any & fparams, std::any & params) {
  data.emplace("new_format", nullptr);
};

std::any kernel_f(unordered_map<string, Format*> & data, std::any & fparams, std::any & pparams, std::any & kparams) {
    return {};
};


TEST(experiment, Basics){
  ConcreteExperiment exp;
  exp.AddDataLoader(load_data_f, {});
  exp.AddPreprocess("id", preprocess_f, {});
  exp.Run(2, true);
  auto aux = exp.GetAuxiliary();
  for(const auto & [id, data]: aux){
    cout << id << endl;

  }
}
