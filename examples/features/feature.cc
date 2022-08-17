#include <iostream>
#include <sparsebase.h>

using id_type = unsigned int;
using nnz_type = unsigned int;
using value_type = float;
using feature_type = unsigned long long int;

using namespace std;
using namespace sparsebase;

int main(int argc, char* argv[]){

  if (argc < 2) {
    cout << "Usage: ./sparse_reader <matrix_market_format>\n";
    cout << "Hint: You can use the matrix market file: "
            "examples/data/ash958.mtx\n";
    return 1;
  }
  string file_name = argv[1];

  /*
  float alpha = 1.0, beta = 0.5;

  sparsebase::preprocess::NumSlices<vertex_type, edge_type, value_type, feature_type> slices;
  auto num_slices =  slices.GetNumSlices(coo, {&cpu_context});

  cout << num_slices[0] << '\n';

  sparsebase::preprocess::NumFibers<vertex_type, edge_type, value_type, feature_type> fibers;
  auto num_fibers =  fibers.GetNumFibers(coo, {&cpu_context});

  cout << num_fibers[0] << '\n';
  */

  sparsebase::format::DimensionType a[3] = {5, 5, 5};
  id_type* b[3];
  b[0]= new id_type[3] {2,3,4};
  b[1]= new id_type[3] {5,6,7};
  b[2]= new id_type[3] {8,9,10};

  id_type ** x = (id_type**) b;

  cout << x[0][0];

  cout << b[0][0];
  value_type c[3] = {3,4,5};
  sparsebase::format::HigherOrderCOO<id_type, nnz_type, value_type>* coo_tensor = new sparsebase::format::HigherOrderCOO<id_type, nnz_type, value_type>(3, ( sparsebase::format::DimensionType* )a, 3, b, (value_type*) c, sparsebase::format::kOwned);

  cout  << '\n' << coo_tensor->get_indices()[1][2] << '\n';

  cout << endl << "TEST 2: Reader" << endl;
  sparsebase::utils::io::TNSReader<id_type, nnz_type, value_type> reader(file_name);
  sparsebase::format::HigherOrderCOO<id_type, nnz_type, value_type>* coo_read_tensor = reader.ReadHigherOrderCOO();

  cout << "Reading successful." << endl;

  cout << "Dimensions : ";
  std::vector<DimensionType> dimension = coo_read_tensor->get_dimensions();

  for(int i=0; i<coo_read_tensor->get_order(); i++)
    cout << dimension[i] << " ";
  cout << endl;

  sparsebase::context::CPUContext cpu_context;

  // NumFibers
  sparsebase::preprocess::NumFibers<id_type, nnz_type, value_type, feature_type> num_fibers;
  feature_type num_fibers_feature = num_fibers.GetNumFibers(coo_read_tensor, {&cpu_context});

  cout << "Num Fibers: " << num_fibers_feature << endl;

  // NumSlices
  sparsebase::preprocess::NumSlices<id_type, nnz_type, value_type, feature_type> num_slices;
  feature_type num_slices_feature = num_slices.GetNumSlices(coo_read_tensor, {&cpu_context});

  cout << "Num Slices: " << num_slices_feature << endl;

  // NnzPerFiber
  sparsebase::preprocess::NnzPerFiber<id_type, nnz_type, value_type, feature_type> nnz_per_fiber;
  nnz_type * nnz_per_fiber_feature = nnz_per_fiber.GetNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "Nnz Per Fiber: " << endl;

  // NnzPerSlice
  sparsebase::preprocess::NnzPerSlice<id_type, nnz_type, value_type, feature_type> nnz_per_slice;
  nnz_type * nnz_per_slice_feature = nnz_per_slice.GetNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "Nnz Per Slice: " << endl;

  // NumNnzFibers
  sparsebase::preprocess::NumNnzFibers<id_type, nnz_type, value_type, feature_type> num_nnz_fibers;
  feature_type num_nnz_fibers_feature = num_nnz_fibers.GetNumNnzFibers(coo_read_tensor, {&cpu_context});

  cout << "Num Nnz Fibers: " << num_nnz_fibers_feature << endl;

  // NumNnzSlices
  sparsebase::preprocess::NumNnzSlices<id_type, nnz_type, value_type, feature_type> num_nnz_slices;
  feature_type num_nnz_slices_feature = num_nnz_slices.GetNumNnzSlices(coo_read_tensor, {&cpu_context});

  cout << "Num Nnz Slices: " << num_nnz_slices_feature << endl;

  cout << endl << endl <<  "Slice related: " << endl;

  // MaxNnzPerSlice
  sparsebase::preprocess::MaxNnzPerSlice<id_type, nnz_type, value_type> max_nnz_per_slice;
  nnz_type max_nnz_per_slice_feature = max_nnz_per_slice.GetMaxNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "MaxNnzPerSlice: " << max_nnz_per_slice_feature << endl;

  // MinNnzPerSlice
  sparsebase::preprocess::MinNnzPerSlice<id_type, nnz_type, value_type> min_nnz_per_slice;
  nnz_type min_nnz_per_slice_feature = min_nnz_per_slice.GetMinNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "MinNnzPerSlice: " << min_nnz_per_slice_feature << endl;

  // AvgNnzPerSlice
  sparsebase::preprocess::AvgNnzPerSlice<id_type, nnz_type, value_type, double> avg_nnz_per_slice;
  double avg_nnz_per_slice_feature = avg_nnz_per_slice.GetAvgNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "AvgNnzPerSlice: " << avg_nnz_per_slice_feature << endl;

  // DevNnzPerSlice
  sparsebase::preprocess::DevNnzPerSlice<id_type, nnz_type, value_type> dev_nnz_per_slice;
  nnz_type dev_nnz_per_slice_feature = dev_nnz_per_slice.GetDevNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "DevNnzPerSlice: " << dev_nnz_per_slice_feature << endl;

  // StdNnzPerSlice
  sparsebase::preprocess::StdNnzPerSlice<id_type, nnz_type, value_type, double> std_nnz_per_slice;
  double std_nnz_per_slice_feature = std_nnz_per_slice.GetStdNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "StdNnzPerSlice: " << std_nnz_per_slice_feature << endl;

  // CovNnzPerSlice
  sparsebase::preprocess::CovNnzPerSlice<id_type, nnz_type, value_type, double> cov_nnz_per_slice;
  double cov_nnz_per_slice_feature = cov_nnz_per_slice.GetCovNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "CovNnzPerSlice: " << cov_nnz_per_slice_feature << endl;

  // AdjNnzPerSlice
  sparsebase::preprocess::AdjNnzPerSlice<id_type, nnz_type, value_type, double> adj_nnz_per_slice;
  double adj_nnz_per_slice_feature = adj_nnz_per_slice.GetAdjNnzPerSlice(coo_read_tensor, {&cpu_context});

  cout << "AdjNnzPerSlice: " << adj_nnz_per_slice_feature << endl;

  cout << endl << endl <<  "Fiber related: " << endl;

  // MaxNnzPerFiber
  sparsebase::preprocess::MaxNnzPerFiber<id_type, nnz_type, value_type> max_nnz_per_fiber;
  nnz_type max_nnz_per_fiber_feature = max_nnz_per_fiber.GetMaxNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "MaxNnzPerFiber: " << max_nnz_per_fiber_feature << endl;

  // MinNnzPerFiber
  sparsebase::preprocess::MinNnzPerFiber<id_type, nnz_type, value_type> min_nnz_per_fiber;
  nnz_type min_nnz_per_fiber_feature = min_nnz_per_fiber.GetMinNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "MinNnzPerFiber: " << min_nnz_per_fiber_feature << endl;

  // AvgNnzPerFiber
  sparsebase::preprocess::AvgNnzPerFiber<id_type, nnz_type, value_type, double> avg_nnz_per_fiber;
  double avg_nnz_per_fiber_feature = avg_nnz_per_fiber.GetAvgNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "AvgNnzPerFiber: " << avg_nnz_per_fiber_feature << endl;

  // DevNnzPerFiber
  sparsebase::preprocess::DevNnzPerFiber<id_type, nnz_type, value_type> dev_nnz_per_fiber;
  nnz_type dev_nnz_per_fiber_feature = dev_nnz_per_fiber.GetDevNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "DevNnzPerFiber: " << dev_nnz_per_fiber_feature << endl;

  // StdNnzPerFiber
  sparsebase::preprocess::StdNnzPerFiber<id_type, nnz_type, value_type, double> std_nnz_per_fiber;
  double std_nnz_per_fiber_feature = std_nnz_per_fiber.GetStdNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "StdNnzPerFiber: " << std_nnz_per_fiber_feature << endl;

  // CovNnzPerFiber
  sparsebase::preprocess::CovNnzPerFiber<id_type, nnz_type, value_type, double> cov_nnz_per_fiber;
  double cov_nnz_per_fiber_feature = cov_nnz_per_fiber.GetCovNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "CovNnzPerFiber: " << cov_nnz_per_fiber_feature << endl;

  // AdjNnzPerFiber
  sparsebase::preprocess::AdjNnzPerFiber<id_type, nnz_type, value_type, double> adj_nnz_per_fiber;
  double adj_nnz_per_fiber_feature = adj_nnz_per_fiber.GetAdjNnzPerFiber(coo_read_tensor, {&cpu_context});

  cout << "AdjNnzPerFiber: " << adj_nnz_per_fiber_feature << endl;

  // MaxFibersPerSlice
  sparsebase::preprocess::MaxFibersPerSlice<id_type, nnz_type, value_type> max_fibers_per_slice;
  nnz_type max_fibers_per_slice_feature = max_fibers_per_slice.GetMaxFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << endl << endl <<  "Fibers Per Slice related: " << endl;

  cout << "MaxFibersPerSlice: " << max_fibers_per_slice_feature << endl;

  // MinFibersPerSlice
  sparsebase::preprocess::MinFibersPerSlice<id_type, nnz_type, value_type> min_fibers_per_slice;
  nnz_type min_fibers_per_slice_feature = min_fibers_per_slice.GetMinFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "MinFibersPerSlice: " << min_fibers_per_slice_feature << endl;

  // AvgFibersPerSlice
  sparsebase::preprocess::AvgFibersPerSlice<id_type, nnz_type, value_type, double> avg_fibers_per_slice;
  double avg_fibers_per_slice_feature = avg_fibers_per_slice.GetAvgFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "AvgFibersPerSlice: " << avg_fibers_per_slice_feature << endl;

  // DevFibersPerSlice
  sparsebase::preprocess::DevFibersPerSlice<id_type, nnz_type, value_type> dev_fibers_per_slice;
  nnz_type dev_fibers_per_slice_feature = dev_fibers_per_slice.GetDevFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "DevFibersPerSlice: " << dev_fibers_per_slice_feature << endl;

  // StdFibersPerSlice
  sparsebase::preprocess::StdFibersPerSlice<id_type, nnz_type, value_type, double> std_fibers_per_slice;
  double std_fibers_per_slice_feature = std_fibers_per_slice.GetStdFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "StdFibersPerSlice: " << std_fibers_per_slice_feature << endl;

  // CovFibersPerSlice
  sparsebase::preprocess::CovFibersPerSlice<id_type, nnz_type, value_type, double> cov_fibers_per_slice;
  double cov_fibers_per_slice_feature = cov_fibers_per_slice.GetCovFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "CovFibersPerSlice: " << cov_fibers_per_slice_feature << endl;

  // AdjFibersPerSlice
  sparsebase::preprocess::AdjFibersPerSlice<id_type, nnz_type, value_type, double> adj_fibers_per_slice;
  double adj_fibers_per_slice_feature = adj_fibers_per_slice.GetAdjFibersPerSlice(coo_read_tensor, {&cpu_context});

  cout << "AdjFibersPerSlice: " << adj_fibers_per_slice_feature << endl;

  return 0;
}
