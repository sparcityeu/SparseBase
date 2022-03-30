#include <iostream>
#include "sparsebase/sparse_format.h"
#include "sparsebase/sparse_reader.h"
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_feature.h"

using namespace std;
using namespace sparsebase;
using namespace format;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = float;
using feature_type = double;

using degrees = preprocess::Degrees<vertex_type, edge_type, value_type>;
using degree_dist = preprocess::DegreeDistribution<vertex_type, edge_type, value_type, feature_type>;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./sparse_feature <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: examples/data/ash958.mtx\n";
    return 1;
  }

  string file_name = argv[1];
  sparsebase::utils::MTXReader<vertex_type, edge_type, value_type> reader(file_name);
  COO<vertex_type, edge_type, value_type> * coo = reader.ReadCOO();
  context::CPUContext cpu_context;

  {
    sparsebase::feature::Extractor engine = sparsebase::feature::FeatureExtractor<vertex_type, edge_type, value_type, feature_type>();
    engine.PrintFuncList();
    engine.Add(feature::Feature(degrees{}));
    engine.Subtract(feature::Feature(degrees{}));
    try{
      engine.Add(feature::Feature(sparsebase::preprocess::DegreeDistribution<vertex_type, edge_type, value_type, float>{}));
    }
    catch (utils::FeatureException & ex){
      cout << ex.what() << endl;
    }
    engine.Add(feature::Feature(degrees{}));
    engine.Add(feature::Feature(degree_dist{}));

    // print features to be extracted
    auto fs = engine.GetList();
    cout << endl << "Features that will be extracted: " << endl;
    for(auto f : fs){
      cout << f.name() << endl;
    }
    cout << endl;

    // extract features
    auto raws = engine.Extract(coo, {&cpu_context});
    cout << "#features extracted: " << raws.size() << endl;
    auto dgrs = std::any_cast<vertex_type*>(raws[degrees::get_feature_id_static()]);
    auto dst = std::any_cast<feature_type*>(raws[degree_dist::get_feature_id_static()]);
    cout << "vertex 0 => degree: " << dgrs[2] << endl;
    cout << "dst[0] " << dst[2] << endl;
  }
  return 0;
}
