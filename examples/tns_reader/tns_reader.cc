#include <iostream>
#include "sparsebase.h"

using id_type = unsigned int;
using nnz_type = unsigned int;
using value_type = float;
using feature_type = unsigned long long int;

using namespace std;
using namespace sparsebase;

int main(int argc, char* argv[]){

  if (argc < 2) {
    cout << "Usage: ./sparse_reader <tns_format>\n";
    cout << "Hint: You can use the tns  file: "
            "examples/data/small.tns\n";
    return 1;
  }
  string file_name = argv[1];

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

  return 0;
}