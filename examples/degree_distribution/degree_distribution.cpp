#include <iostream>
#include "sparsebase/sparse_preprocess.h"
#include "sparsebase/sparse_object.h"

using namespace std;
using namespace sparsebase;

using vertex_type = unsigned int;
using edge_type = unsigned int;
using value_type = unsigned int;
using feature_type = float;

int main(int argc, char * argv[]){
  if (argc < 2){
    cout << "Usage: ./degree_distribution <uedgelist_file>\n";
    cout << "Hint: You can use the edgelist: examples/data/com-dblp.uedgelist\n";
    return 1;
  }
  cout << "F t re  s sp r e!" << endl;
  string file_name = argv[1];

  cout << "********************************" << endl;

  cout << "Reading graph from " << file_name << "..." << endl;
  object::Graph<vertex_type, edge_type, value_type> g;
  g.ReadConnectivityFromEdgelistToCSR(file_name);
  cout << "Number of vertices: " << g.n_ << endl; 
  cout << "Number of edges: " << g.m_ << endl; 

  cout << "********************************" << endl;
  cout << "Calculating degree distribution..." << endl;

  sparsebase::preprocess::DegreeDistribution<vertex_type, edge_type, value_type, feature_type> extractor;
  //format::SparseFormat<vertex_type, edge_type, value_type> * con = g.get_connectivity();
  //feature_type * degree_distribution = extractor.GetFeatures(con);
  feature_type * degree_distribution = extractor.GetDistribution(g.get_connectivity());
  feature_type * degree_distribution2 = extractor.GetDistribution(&g);

  cout << degree_distribution[0] << endl;
  cout << degree_distribution2[0] << endl;


  return 0;
}