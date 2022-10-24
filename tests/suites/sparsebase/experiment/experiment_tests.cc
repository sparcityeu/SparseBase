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

const ConcreteExperiment experiment;
const string FILE_NAME = "../../../../examples/data/ash958.mtx";

auto load_data_f( vector<string> & file_names ) {
  CSR<vertex_type, edge_type, value_type> * csr = MTXReader<vertex_type, edge_type, value_type>(FILE_NAME)
      .ReadCSR();
  unordered_map<string, Format*> r;
  r.emplace("format", csr);
  return r;
};

void preprocess_f(unordered_map<string, Format*> & data, std::any & params) {

  data.emplace("ordered", A_reordered);
};

std::any kernel_f(unordered_map<string, Format*> & data, std::any & fparams, std::any & pparams, std::any & kparams) {
};


TEST(experiment, Basics){
 cout << "tstss" << endl;
}
