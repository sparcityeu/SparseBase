#include "sparsebase/feature/feature.h"
#include "sparsebase/format/format.h"
#include "sparsebase/preprocess/preprocess.h"
#include "sparsebase/utils/io/reader.h"
#include <iostream>

using namespace std;
using namespace sparsebase;
using namespace format;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = float;
using feature_type = double;

using degrees = preprocess::Degrees<vertex_type, edge_type, value_type>;
using degree_dist = preprocess::DegreeDistribution<vertex_type, edge_type,
    value_type, feature_type>;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: ./sparse_feature <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: "
            "examples/data/ash958.mtx\n";
    return 1;
  }

  // The name of the matrix-market file
  string file_name = argv[1];
  // Initialize a reader object with the matrix-market file inputted
  sparsebase::utils::io::MTXReader<vertex_type, edge_type, value_type> reader(
      file_name);
  // Read the matrix in to a COO representation
  COO<vertex_type, edge_type, value_type> *coo = reader.ReadCOO();

  // Create an extractor with the correct types of your COO (data) and your expected feature type
  sparsebase::feature::Extractor engine =
      sparsebase::feature::FeatureExtractor<vertex_type, edge_type,
          value_type, feature_type>();

  //add all the feature you want to extract to the extractor
  engine.Add(feature::Feature(degrees{}));
  engine.Add(feature::Feature(degree_dist{}));

  // print features to be extracted
  auto fs = engine.GetList();
  cout << endl << "Features that will be extracted: " << endl;
  for (auto f : fs) {
    cout << f.name() << endl;
  }
  cout << endl;

  // Create a context, CPUcontext for this case.
  // The contexts defines the architecture that the computation will take place in.
  context::CPUContext cpu_context;
  // extract features
  auto raws = engine.Extract(coo, {&cpu_context});

  cout << "#features extracted: " << raws.size() << endl;
  auto dgrs =
      std::any_cast<vertex_type *>(raws[degrees::get_feature_id_static()]);
  auto dst = std::any_cast<feature_type *>(
      raws[degree_dist::get_feature_id_static()]);
  cout << "vertex 0 => degree: " << dgrs[2] << endl;
  cout << "dst[0] " << dst[2] << endl;
  return 0;
}
